"""
Horizon Ledger — Daily Price Fetching
Downloads OHLCV data via yfinance and stores in daily_prices table.
Also fetches benchmark (SPY) prices.
"""

import concurrent.futures
import logging
import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from config import PRICE_HISTORY_YEARS, BENCHMARK_TICKER
from db.schema import get_connection
from db.queries import get_universe_tickers, get_stock_id, upsert_prices, upsert_stock

log = logging.getLogger(__name__)

BATCH_SIZE = 50    # Smaller batches are less likely to hang; 50 is reliable with yfinance
_DOWNLOAD_TIMEOUT = 90  # seconds per batch before giving up and moving on


def _download_with_timeout(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Wrap yf.download in a thread so we can enforce a hard timeout.
    Raises concurrent.futures.TimeoutError if the download stalls.
    """
    tickers_str = " ".join(tickers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(
            yf.download,
            tickers_str,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        return future.result(timeout=_DOWNLOAD_TIMEOUT)


def _to_str_date(d) -> str:
    return d.isoformat() if isinstance(d, date) else str(d)


def _store_batch(conn, batch: list[str], raw: pd.DataFrame, results: dict) -> None:
    """Parse a yf.download() result and upsert into daily_prices."""
    with conn:
        for tkr in batch:
            try:
                if len(batch) == 1:
                    df = raw.copy()
                    # Newer yfinance may return MultiIndex columns even for a single ticker.
                    # Flatten to get plain 'Close', 'Open', etc.
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                else:
                    if tkr not in raw.columns.get_level_values(0):
                        log.debug("No data returned for %s", tkr)
                        continue
                    df = raw[tkr].copy()
                    # Ticker-slice may still have MultiIndex in edge cases
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                df = df.dropna(subset=["Close"])
                if df.empty:
                    continue

                df = df.reset_index()
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]

                col_map = {
                    "datetime": "date", "timestamp": "date",
                    "adj_close": "adj_close", "close": "adj_close",
                }
                df = df.rename(columns=col_map)
                if "adj_close" not in df.columns and "close" in df.columns:
                    df["adj_close"] = df["close"]

                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

                for col in ["open", "high", "low", "close"]:
                    if col not in df.columns:
                        df[col] = df.get("adj_close")

                stock_id = get_stock_id(conn, tkr)
                if stock_id is None:
                    upsert_stock(conn, tkr, is_active=1)
                    stock_id = get_stock_id(conn, tkr)

                n = upsert_prices(conn, stock_id, df)
                results[tkr] = n
            except Exception as e:
                log.warning("Price store failed for %s: %s", tkr, e)
        conn.commit()


def fetch_and_store_prices(
    tickers: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    full_history: bool = False,
) -> dict[str, int]:
    """
    Download OHLCV for a list of tickers and upsert into daily_prices.
    Returns dict of {ticker: rows_upserted}.
    """
    if start is None:
        if full_history:
            start = (
                date.today() - timedelta(days=PRICE_HISTORY_YEARS * 365)
            ).isoformat()
        else:
            # Default: last 7 days (daily update)
            start = (date.today() - timedelta(days=7)).isoformat()
    if end is None:
        end = date.today().isoformat()

    results: dict[str, int] = {}
    conn = get_connection()

    # Resume support: skip tickers that already have recent data in the DB.
    # A ticker is considered "already loaded" if it has ≥200 price rows,
    # which implies a multi-year history is present.
    if full_history:
        already_loaded = set(
            row[0]
            for row in conn.execute(
                """
                SELECT s.ticker
                FROM stocks s
                JOIN daily_prices dp ON dp.stock_id = s.id
                GROUP BY s.ticker
                HAVING COUNT(*) >= 200
                """
            ).fetchall()
        )
        skipped = [t for t in tickers if t in already_loaded]
        tickers  = [t for t in tickers if t not in already_loaded]
        if skipped:
            log.info(
                "  Resume: skipping %d tickers that already have full history "
                "(%d remaining to fetch)",
                len(skipped), len(tickers),
            )

    n_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        log.info(
            "Fetching prices batch %d/%d (%d tickers)...",
            batch_num, n_batches, len(batch),
        )

        raw = None
        for attempt in (1, 2):  # retry once on timeout
            try:
                raw = _download_with_timeout(batch, start, end)
                break
            except concurrent.futures.TimeoutError:
                log.warning(
                    "Batch %d/%d timed out after %ds (attempt %d/2) — %s",
                    batch_num, n_batches, _DOWNLOAD_TIMEOUT, attempt,
                    "retrying with smaller sub-batches..." if attempt == 1 else "skipping batch",
                )
                if attempt == 1:
                    # Retry as two half-batches to isolate the stuck ticker
                    mid = len(batch) // 2
                    sub_results = {}
                    for sub in (batch[:mid], batch[mid:]):
                        if not sub:
                            continue
                        try:
                            sub_raw = _download_with_timeout(sub, start, end)
                            sub_results[tuple(sub)] = sub_raw
                        except Exception as sub_e:
                            log.warning("Sub-batch also failed: %s", sub_e)
                    # Process each successful sub-batch immediately
                    for sub_tickers, sub_raw in sub_results.items():
                        _store_batch(conn, list(sub_tickers), sub_raw, results)
                    raw = None  # signal that we already stored
                    break
            except Exception as e:
                log.error("Batch %d/%d download error: %s", batch_num, n_batches, e)
                break

        if raw is not None:
            _store_batch(conn, batch, raw, results)

        time.sleep(0.5)  # small pause between batches to be polite to Yahoo

    conn.close()
    total = sum(results.values())
    log.info("Prices stored: %d rows across %d tickers", total, len(results))
    return results


def update_daily_prices() -> None:
    """Incremental daily update: fetch last 7 days for all active tickers."""
    conn = get_connection()
    tickers = get_universe_tickers(conn)
    conn.close()

    # Always include benchmark
    if BENCHMARK_TICKER not in tickers:
        tickers.append(BENCHMARK_TICKER)

    log.info("Daily price update for %d tickers...", len(tickers))
    fetch_and_store_prices(tickers, full_history=False)


def backfill_prices(tickers: Optional[list[str]] = None) -> None:
    """
    One-time backfill: download full PRICE_HISTORY_YEARS of history.
    ⚠️ Survivorship bias warning: downloads only for current universe members.
    """
    log.warning(
        "SURVIVORSHIP BIAS WARNING: backfill uses current universe only. "
        "Historical analysis will be biased toward survivors."
    )
    if tickers is None:
        conn = get_connection()
        tickers = get_universe_tickers(conn)
        conn.close()

    if BENCHMARK_TICKER not in tickers:
        tickers.append(BENCHMARK_TICKER)

    log.info("Backfilling %d years of prices for %d tickers...", PRICE_HISTORY_YEARS, len(tickers))
    fetch_and_store_prices(tickers, full_history=True)


def get_price_series(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    field: str = "adj_close",
) -> pd.Series:
    """
    Convenience function: return a price series from the DB.
    Falls back to yfinance if not in DB.
    """
    conn = get_connection()
    stock_id = get_stock_id(conn, ticker)
    if stock_id:
        from db.queries import get_prices
        df = get_prices(conn, stock_id, start, end)
        conn.close()
        if not df.empty and field in df.columns:
            return df.set_index("date")[field]
    conn.close()
    # Fallback: yfinance live fetch
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if not raw.empty:
            s = raw["Close"]
            s.index = s.index.strftime("%Y-%m-%d")
            s.name = ticker
            return s
    except Exception as e:
        log.warning("Fallback yfinance failed for %s: %s", ticker, e)
    return pd.Series(dtype=float, name=ticker)


def compute_forward_return(
    ticker: str,
    signal_date: str,
    holding_days: int,
) -> Optional[float]:
    """
    Compute the price-only total return from signal_date to signal_date + holding_days.
    Returns None if data is unavailable.
    """
    from db.queries import get_price_on_date

    conn = get_connection()
    stock_id = get_stock_id(conn, ticker)
    if stock_id is None:
        conn.close()
        return None

    entry_price = get_price_on_date(conn, stock_id, signal_date)
    exit_date = (
        date.fromisoformat(signal_date) + timedelta(days=holding_days)
    ).isoformat()
    exit_price = get_price_on_date(conn, stock_id, exit_date)
    conn.close()

    if entry_price and exit_price and entry_price > 0:
        return (exit_price - entry_price) / entry_price
    return None
