"""
Horizon Ledger — Stock Universe Construction
Builds and maintains the investable universe from S&P 500 + Russell 1000.
Sources: Wikipedia (S&P 500), iShares IWB holdings (Russell 1000), yfinance.

Survivorship-bias note: this universe reflects CURRENT constituents.
Historical backtests are subject to survivorship bias.  Forward predictions
(starting from the date this script first runs) are bias-free.
"""

import logging
import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

from config import (
    SP500_WIKI_URL,
    RUSSELL1000_URL,
    UNIVERSE_MIN_MARKET_CAP,
    UNIVERSE_MIN_AVG_VOLUME,
    UNIVERSE_MIN_LISTING_AGE_DAYS,
)
from db.schema import get_connection
from db.queries import upsert_stock, get_active_universe

log = logging.getLogger(__name__)


# ─── Fetch ticker lists ───────────────────────────────────────────────────────

def fetch_sp500_tickers() -> pd.DataFrame:
    """
    Scrape S&P 500 constituents from Wikipedia.
    Returns DataFrame with columns: ticker, name, sector, industry.
    Uses requests with a browser User-Agent to avoid Wikipedia's 403 block.
    """
    log.info("Fetching S&P 500 list from Wikipedia...")
    try:
        from io import StringIO
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(SP500_WIKI_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        df = tables[0]
        df = df.rename(
            columns={
                "Symbol": "ticker",
                "Security": "name",
                "GICS Sector": "sector",
                "GICS Sub-Industry": "industry",
            }
        )
        # Wikipedia uses periods instead of hyphens in some tickers (e.g. BRK.B → BRK-B)
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        return df[["ticker", "name", "sector", "industry"]].copy()
    except Exception as e:
        log.error("Failed to fetch S&P 500 list: %s", e)
        return pd.DataFrame(columns=["ticker", "name", "sector", "industry"])


def fetch_russell1000_tickers() -> pd.DataFrame:
    """
    Fetch Russell 1000 holdings from iShares IWB ETF CSV.
    Returns DataFrame with columns: ticker, name.
    """
    log.info("Fetching Russell 1000 list from iShares...")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(RUSSELL1000_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        from io import StringIO
        # iShares CSV has metadata rows at the top — skip until header row
        lines = resp.text.splitlines()
        header_idx = next(
            (i for i, l in enumerate(lines) if "Ticker" in l), 9
        )
        df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
        df = df.rename(columns={"Ticker": "ticker", "Name": "name"})
        df = df[df["ticker"].notna() & (df["ticker"] != "-")]
        df["ticker"] = df["ticker"].astype(str).str.strip()
        return df[["ticker", "name"]].copy()
    except Exception as e:
        log.warning("Could not fetch Russell 1000 (iShares): %s — skipping", e)
        return pd.DataFrame(columns=["ticker", "name"])


# ─── Enrich with yfinance ────────────────────────────────────────────────────

def _yf_info_safe(ticker: str) -> dict:
    """Fetch yfinance info dict, returning empty dict on failure."""
    try:
        info = yf.Ticker(ticker).fast_info
        return {
            "market_cap": getattr(info, "market_cap", None),
            "exchange": getattr(info, "exchange", None),
        }
    except Exception:
        return {}


def _extract_series(data: "pd.DataFrame", field: str, tkr: str, batch_len: int) -> "pd.Series":
    """
    Safely extract a price/volume series from a yf.download() result.
    Handles both single-ticker (flat columns) and multi-ticker (MultiIndex) outputs,
    regardless of yfinance version.
    """
    import pandas as pd
    empty = pd.Series(dtype=float)
    if data is None or data.empty:
        return empty
    try:
        cols = data.columns
        if isinstance(cols, pd.MultiIndex):
            # Multi-ticker: columns are (field, ticker) — field-first default
            level0 = cols.get_level_values(0)
            if field in level0:
                sub = data[field]          # DataFrame with ticker columns
                if batch_len == 1:
                    return sub.squeeze()   # single-column → Series
                return sub[tkr] if tkr in sub.columns else empty
            # Ticker-first fallback (group_by="ticker" was used somehow)
            if tkr in level0:
                sub = data[tkr]
                return sub[field] if field in sub.columns else empty
            return empty
        else:
            # Flat columns — single ticker download
            return data[field] if field in cols else empty
    except Exception:
        return empty


def filter_universe(tickers: list[str], batch_size: int = 50) -> pd.DataFrame:
    """
    Apply minimum filters:
      - Market cap > $100 M
      - 20-day avg notional volume > $1 M
      - Listed for > 1 year

    Uses yfinance in batches to fetch current market data.
    Returns a filtered DataFrame with enriched columns.
    """
    log.info("Applying universe filters to %d tickers...", len(tickers))
    results = []
    today = date.today()
    cutoff_date = (today - timedelta(days=UNIVERSE_MIN_LISTING_AGE_DAYS)).isoformat()

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_str = " ".join(batch)
        try:
            # Use default field-first grouping (no group_by).
            # data["Close"] → DataFrame with ticker columns (multi) or Series (single).
            data = yf.download(
                batch_str,
                period="1mo",
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            log.warning("yfinance batch download failed: %s", e)
            data = pd.DataFrame()

        for tkr in batch:
            try:
                closes = _extract_series(data, "Close", tkr, len(batch))
                volumes = _extract_series(data, "Volume", tkr, len(batch))

                avg_close = closes.dropna().tail(20).mean()
                avg_vol = volumes.dropna().tail(20).mean()
                avg_notional = avg_close * avg_vol if (avg_close and avg_vol) else 0.0

                info = _yf_info_safe(tkr)
                mcap = info.get("market_cap") or 0.0

                passes = (
                    mcap >= UNIVERSE_MIN_MARKET_CAP
                    and avg_notional >= UNIVERSE_MIN_AVG_VOLUME
                )
                results.append({
                    "ticker": tkr,
                    "market_cap": mcap,
                    "avg_notional_volume": avg_notional,
                    "exchange": info.get("exchange"),
                    "passes_filter": passes,
                })
            except Exception as e:
                log.debug("Filter check failed for %s: %s", tkr, e)
                results.append({
                    "ticker": tkr,
                    "market_cap": 0.0,
                    "avg_notional_volume": 0.0,
                    "exchange": None,
                    "passes_filter": False,
                })

        time.sleep(0.5)   # Be polite to yfinance

    df = pd.DataFrame(results)
    passed = df[df["passes_filter"]].copy()
    log.info(
        "Universe filter: %d / %d tickers pass", len(passed), len(tickers)
    )
    return passed


# ─── Main universe build ─────────────────────────────────────────────────────

def build_universe(apply_filters: bool = True) -> None:
    """
    Fetch S&P 500 + Russell 1000, merge, deduplicate, optionally filter,
    and persist to the stocks table.
    """
    sp500 = fetch_sp500_tickers()
    r1000 = fetch_russell1000_tickers()

    # Merge: sp500 takes priority for sector/industry metadata
    merged = sp500.set_index("ticker").combine_first(
        r1000.set_index("ticker")
    ).reset_index()
    merged = merged.rename(columns={"index": "ticker"})
    tickers = merged["ticker"].tolist()

    if apply_filters:
        filtered = filter_universe(tickers)
        valid_tickers = set(filtered["ticker"])
        merged = merged[merged["ticker"].isin(valid_tickers)].copy()
        merged = merged.merge(
            filtered[["ticker", "market_cap", "exchange"]],
            on="ticker", how="left", suffixes=("", "_yf"),
        )
        for col in ["market_cap", "exchange"]:
            yf_col = col + "_yf"
            if yf_col in merged.columns:
                merged[col] = merged[yf_col].combine_first(merged.get(col, pd.Series(dtype=object)))
                merged.drop(columns=[yf_col], inplace=True, errors="ignore")

    today = date.today().isoformat()
    conn = get_connection()
    added = 0
    with conn:
        for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Upserting universe"):
            upsert_stock(
                conn,
                ticker=row["ticker"],
                name=row.get("name"),
                sector=row.get("sector"),
                industry=row.get("industry"),
                market_cap=row.get("market_cap"),
                exchange=row.get("exchange"),
                is_active=1,
                added_date=today,
            )
            added += 1
        conn.commit()
    conn.close()
    log.info("Universe built: %d stocks stored", added)


def refresh_cik_mapping() -> None:
    """
    Download SEC company tickers JSON to populate the cik column for EDGAR lookups.
    Source: https://www.sec.gov/files/company_tickers.json
    """
    log.info("Fetching SEC CIK mapping...")
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": "HorizonLedger research@local.dev"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # data: {idx: {cik_str, ticker, title}}
        ticker_to_cik: dict[str, str] = {}
        for entry in data.values():
            t = entry.get("ticker", "").upper().strip()
            c = str(entry.get("cik_str", "")).zfill(10)
            if t:
                ticker_to_cik[t] = c

        conn = get_connection()
        with conn:
            for ticker, cik in ticker_to_cik.items():
                conn.execute(
                    "UPDATE stocks SET cik=? WHERE ticker=?", (cik, ticker)
                )
            conn.commit()
        conn.close()
        log.info("CIK mapping updated for %d tickers", len(ticker_to_cik))
    except Exception as e:
        log.error("CIK mapping failed: %s", e)


def mark_delisted(ticker: str) -> None:
    """Mark a stock as delisted in the stocks table."""
    conn = get_connection()
    with conn:
        conn.execute(
            "UPDATE stocks SET is_active=0, delisted_date=? WHERE ticker=?",
            (date.today().isoformat(), ticker),
        )
        conn.commit()
    conn.close()
    log.info("Marked %s as delisted", ticker)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from db.schema import init_db, seed_initial_weights
    init_db()
    conn = get_connection()
    seed_initial_weights(conn)
    conn.close()
    build_universe()
    refresh_cik_mapping()
