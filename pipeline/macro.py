"""
Horizon Ledger — Macro Regime Indicators
Fetches FRED series and computes macro regime features.
Free FRED API (120 req/min with key).
"""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import numpy as np

from config import FRED_API_KEY, FRED_SERIES
from db.schema import get_connection
from db.queries import upsert_macro, get_macro_series

log = logging.getLogger(__name__)


def fetch_fred_series(
    series_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.Series:
    """
    Download a single FRED series.  Returns a pandas Series indexed by date string.
    Falls back to yfinance (^VIX) if FRED fails for VIX.
    """
    if start is None:
        start = (date.today() - timedelta(days=365 * 12)).isoformat()
    if end is None:
        end = date.today().isoformat()
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        s.index = s.index.strftime("%Y-%m-%d")
        s.name = series_id
        return s.dropna()
    except Exception as e:
        log.warning("FRED fetch failed for %s: %s", series_id, e)
        if series_id == "VIXCLS":
            return _fetch_vix_yfinance(start, end)
        return pd.Series(dtype=float, name=series_id)


def _fetch_vix_yfinance(start: str, end: str) -> pd.Series:
    """Fallback: fetch VIX from yfinance as ^VIX."""
    try:
        import yfinance as yf
        df = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            return pd.Series(dtype=float, name="VIXCLS")
        s = df["Close"]
        s.index = s.index.strftime("%Y-%m-%d")
        s.name = "VIXCLS"
        return s.dropna()
    except Exception as e:
        log.error("VIX yfinance fallback failed: %s", e)
        return pd.Series(dtype=float, name="VIXCLS")


def update_macro_data(years: int = 12) -> None:
    """Fetch and store all configured FRED series."""
    start = (date.today() - timedelta(days=365 * years)).isoformat()
    conn = get_connection()
    with conn:
        for series_id in FRED_SERIES:
            log.info("Fetching FRED series: %s", series_id)
            s = fetch_fred_series(series_id, start=start)
            if s.empty:
                log.warning("No data for %s", series_id)
                continue
            for date_str, value in s.items():
                if pd.notna(value):
                    upsert_macro(conn, series_id, date_str, float(value))
            log.info("  → %d observations stored for %s", len(s), series_id)
        conn.commit()
    conn.close()


def get_macro_features(as_of: Optional[str] = None) -> dict:
    """
    Compute macro regime features for HMM input.
    Returns dict with: yield_curve_slope, credit_spread, vix_level,
    cpi_yoy, sp500_monthly_return.
    """
    if as_of is None:
        as_of = date.today().isoformat()

    start = (date.fromisoformat(as_of) - timedelta(days=90)).isoformat()
    conn = get_connection()

    def _latest(series_id: str) -> Optional[float]:
        df = get_macro_series(conn, series_id, start=start, end=as_of)
        if df.empty:
            return None
        return df.iloc[-1]["value"]

    dgs10 = _latest("DGS10")
    dgs2  = _latest("DGS2")
    vix   = _latest("VIXCLS")
    hy    = _latest("BAMLH0A0HYM2")

    # CPI year-over-year
    cpi_now = _latest("CPIAUCSL")
    cpi_1y_ago_start = (
        date.fromisoformat(as_of) - timedelta(days=400)
    ).isoformat()
    cpi_1y_ago_end = (
        date.fromisoformat(as_of) - timedelta(days=335)
    ).isoformat()
    cpi_df_old = get_macro_series(conn, "CPIAUCSL", start=cpi_1y_ago_start, end=cpi_1y_ago_end)
    cpi_old = cpi_df_old.iloc[-1]["value"] if not cpi_df_old.empty else None
    cpi_yoy = ((cpi_now / cpi_old) - 1) if (cpi_now and cpi_old and cpi_old > 0) else None

    conn.close()

    yield_curve = (dgs10 - dgs2) if (dgs10 is not None and dgs2 is not None) else None

    # S&P 500 monthly return from prices
    sp500_return = _get_sp500_monthly_return(as_of)

    return {
        "yield_curve_slope": yield_curve,
        "credit_spread": hy,
        "vix_level": vix,
        "cpi_yoy": cpi_yoy,
        "sp500_monthly_return": sp500_return,
        "dgs10": dgs10,
        "dgs2": dgs2,
    }


def _get_sp500_monthly_return(as_of: str) -> Optional[float]:
    """Get SPY 21-day return from DB prices."""
    try:
        from db.schema import get_connection as gc
        from db.queries import get_stock_id, get_price_on_date
        conn = gc()
        spy_id = get_stock_id(conn, "SPY")
        if spy_id is None:
            conn.close()
            return None
        end_price = get_price_on_date(conn, spy_id, as_of)
        start_date = (date.fromisoformat(as_of) - timedelta(days=30)).isoformat()
        start_price = get_price_on_date(conn, spy_id, start_date)
        conn.close()
        if end_price and start_price and start_price > 0:
            return (end_price - start_price) / start_price
    except Exception as e:
        log.debug("SP500 monthly return error: %s", e)
    return None


def get_macro_history_for_hmm(years: int = 10) -> pd.DataFrame:
    """
    Return a monthly DataFrame of macro features for HMM training.
    Columns: date, sp500_monthly_return, yield_curve_slope, credit_spread, vix_level
    """
    start = (date.today() - timedelta(days=365 * years)).isoformat()
    conn = get_connection()

    dgs10 = get_macro_series(conn, "DGS10", start=start)
    dgs2  = get_macro_series(conn, "DGS2",  start=start)
    vix   = get_macro_series(conn, "VIXCLS", start=start)
    hy    = get_macro_series(conn, "BAMLH0A0HYM2", start=start)
    conn.close()

    def _to_monthly(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["date", col_name])
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").resample("ME").last().reset_index()
        df["date"] = df["date"].dt.strftime("%Y-%m")
        df = df.rename(columns={"value": col_name})
        return df

    m_dgs10 = _to_monthly(dgs10, "dgs10")
    m_dgs2  = _to_monthly(dgs2,  "dgs2")
    m_vix   = _to_monthly(vix,   "vix")
    m_hy    = _to_monthly(hy,    "credit_spread")

    df = m_dgs10.merge(m_dgs2, on="date", how="outer")
    df = df.merge(m_vix, on="date", how="outer")
    df = df.merge(m_hy, on="date", how="outer")

    df["yield_curve_slope"] = df["dgs10"] - df["dgs2"]

    # SPY monthly returns
    spy_returns = _get_spy_monthly_returns(start)
    df = df.merge(spy_returns, on="date", how="left")

    df = df.dropna(subset=["yield_curve_slope", "vix"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_cape_data() -> pd.DataFrame:
    """
    Download Robert Shiller's CAPE (Cyclically Adjusted P/E) data from Yale.
    Source: http://www.econ.yale.edu/~shiller/data/ie_data.xls
    Returns a DataFrame with columns: date (YYYY-MM), cape.
    Column name varies across file versions — searched case-insensitively.

    Requires xlrd>=2.0.0 for .xls format support.
    """
    from config import CAPE_URL
    log.info("Downloading Shiller CAPE data from Yale...")
    try:
        resp = requests.get(CAPE_URL, timeout=60, headers={"User-Agent": "HorizonLedger/1.0"})
        resp.raise_for_status()
        from io import BytesIO
        import xlrd  # noqa: F401 — ensures xlrd is available for pd.read_excel
        df = pd.read_excel(BytesIO(resp.content), sheet_name="Data", engine="xlrd", header=7)
        df.columns = [str(c).strip() for c in df.columns]

        # Find the CAPE column — name has changed across Shiller file versions
        cape_col = next(
            (c for c in df.columns
             if any(kw in c.upper() for kw in ["CAPE", "P/E10", "PE10", "CYCLICALLY"])),
            None,
        )
        if cape_col is None:
            log.error("Could not find CAPE column in Shiller data. Columns: %s", df.columns.tolist())
            return pd.DataFrame(columns=["date", "cape"])

        # Date column is usually first column as decimal year (e.g. 1881.01)
        date_col = df.columns[0]
        df = df[[date_col, cape_col]].copy()
        df.columns = ["date_raw", "cape"]
        df = df.dropna(subset=["cape"])
        df = df[pd.to_numeric(df["cape"], errors="coerce").notna()]
        df["cape"] = pd.to_numeric(df["cape"], errors="coerce")

        # Convert decimal year to YYYY-MM string
        def _decimal_to_ym(v):
            try:
                v = float(v)
                year = int(v)
                month = int(round((v - year) * 12)) + 1
                month = max(1, min(12, month))
                return f"{year:04d}-{month:02d}"
            except Exception:
                return None

        df["date"] = df["date_raw"].apply(_decimal_to_ym)
        df = df[df["date"].notna()][["date", "cape"]].reset_index(drop=True)
        log.info("  → Loaded %d CAPE observations (earliest: %s)", len(df), df["date"].iloc[0])
        return df
    except Exception as e:
        log.error("CAPE data download failed: %s", e)
        return pd.DataFrame(columns=["date", "cape"])


def get_cape_stats(conn) -> dict:
    """
    Return current CAPE, its historical percentile, and expected 10y return.
    Reads from macro_data table (series='CAPE').
    """
    from db.queries import get_macro_series
    cape_df = get_macro_series(conn, "CAPE")
    if cape_df.empty:
        return {}
    cape_values = cape_df["value"].dropna()
    current_cape = cape_values.iloc[-1]
    percentile = float((cape_values < current_cape).mean() * 100)

    # Historical 10-year forward return approximation based on CAPE level
    # Source: Shiller et al. — declining returns at higher valuations
    def _expected_10y_return(cape: float) -> float:
        if cape <= 0:
            return 0.0
        if cape < 10:
            return 0.14
        elif cape < 15:
            return 0.11
        elif cape < 20:
            return 0.085
        elif cape < 25:
            return 0.065
        elif cape < 30:
            return 0.045
        elif cape < 35:
            return 0.025
        else:
            return 0.01

    return {
        "cape_ratio": current_cape,
        "cape_percentile": percentile,
        "expected_10y_return": _expected_10y_return(current_cape),
        "historical_25th": float(cape_values.quantile(0.25)),
        "historical_50th": float(cape_values.quantile(0.50)),
        "historical_75th": float(cape_values.quantile(0.75)),
        "historical_90th": float(cape_values.quantile(0.90)),
    }


def update_cape_in_macro_db() -> None:
    """Download Shiller CAPE and store monthly values in macro_data as series='CAPE'."""
    conn = get_connection()
    cape_df = fetch_cape_data()
    if cape_df.empty:
        conn.close()
        return
    with conn:
        for _, row in cape_df.iterrows():
            # Store as YYYY-MM-01 for consistent date format
            date_str = row["date"] + "-01"
            upsert_macro(conn, "CAPE", date_str, float(row["cape"]))
        conn.commit()
    conn.close()
    log.info("CAPE data stored: %d monthly observations", len(cape_df))


def compute_pct_above_200sma() -> Optional[float]:
    """
    Count the percentage of active universe stocks trading above their 200-day SMA.
    Uses the technicals parquet cache if available; falls back to daily_prices table.
    Returns float 0-100, or None if insufficient data.
    """
    from config import DATA_DIR, SMA_LONG
    from db.queries import get_active_universe, get_stock_id
    import glob

    conn = get_connection()
    universe = get_active_universe(conn)
    conn.close()

    if universe.empty:
        return None

    above = 0
    total = 0
    parquet_dir = DATA_DIR / "technicals"

    for _, row in universe.iterrows():
        ticker = row["ticker"]
        try:
            parquet_path = parquet_dir / f"{ticker}.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                if "sma_200" in df.columns and "close" in df.columns and not df.empty:
                    last = df.dropna(subset=["sma_200", "close"]).iloc[-1]
                    if last["close"] > last["sma_200"]:
                        above += 1
                    total += 1
        except Exception:
            pass

    if total < 50:
        # Fallback: compute directly from daily_prices
        conn = get_connection()
        try:
            from db.queries import get_stock_id, get_prices
            above = 0
            total = 0
            for _, row in universe.iterrows():
                sid = get_stock_id(conn, row["ticker"])
                if sid is None:
                    continue
                prices = get_prices(conn, sid)
                if prices.empty or len(prices) < SMA_LONG:
                    continue
                prices = prices.sort_values("date")
                prices["sma_200"] = prices["adj_close"].rolling(SMA_LONG).mean()
                last = prices.dropna(subset=["sma_200"]).iloc[-1]
                if last["adj_close"] > last["sma_200"]:
                    above += 1
                total += 1
        finally:
            conn.close()

    if total == 0:
        return None
    return round(above / total * 100, 1)


def _get_spy_monthly_returns(start: str) -> pd.DataFrame:
    """Compute monthly SPY returns from daily_prices table."""
    try:
        from db.schema import get_connection as gc
        from db.queries import get_stock_id, get_prices
        conn = gc()
        spy_id = get_stock_id(conn, "SPY")
        if spy_id is None:
            conn.close()
            return pd.DataFrame(columns=["date", "sp500_monthly_return"])
        prices = get_prices(conn, spy_id, start=start)
        conn.close()
        if prices.empty:
            return pd.DataFrame(columns=["date", "sp500_monthly_return"])
        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.set_index("date").sort_index()
        monthly = prices["adj_close"].resample("ME").last()
        returns = monthly.pct_change().dropna()
        result = returns.reset_index()
        result["date"] = result["date"].dt.strftime("%Y-%m")
        result = result.rename(columns={"adj_close": "sp500_monthly_return"})
        return result
    except Exception as e:
        log.warning("SPY monthly returns error: %s", e)
        return pd.DataFrame(columns=["date", "sp500_monthly_return"])
