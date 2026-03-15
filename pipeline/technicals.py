"""
Horizon Ledger — Technical Indicator Computation
Uses pandas-ta-classic to compute indicators from daily price data.
Results are stored as an in-memory/file cache (not in the main SQLite DB).

Indicators computed:
  SMA_50, SMA_200, EMA_20
  RSI_14, RSI_5
  MACD(12,26,9)
  Bollinger Bands(20, 2.0)
  ATR_14
  OBV
  Volume_SMA_20
  Golden/Death cross detection
  RSI divergence, MACD divergence
  Breakout volume flag
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    BASE_DIR,
    SMA_SHORT, SMA_LONG, EMA_SHORT,
    RSI_PERIOD, RSI_SHORT,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD,
    ATR_PERIOD, VOL_SMA,
    DIVERGENCE_WINDOW, BREAKOUT_VOLUME_MULT,
)

log = logging.getLogger(__name__)

CACHE_DIR = BASE_DIR / "data" / "technicals"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators on a price DataFrame.
    Input df must have columns: date, open, high, low, close/adj_close, volume
    Returns df with indicator columns added.
    """
    if df.empty:
        return df

    df = df.copy()
    if "adj_close" in df.columns:
        df["close"] = df["adj_close"]
    elif "close" not in df.columns:
        log.warning("No close price column found")
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    close = df["close"]
    high  = df["high"] if "high" in df.columns else close
    low   = df["low"]  if "low"  in df.columns else close
    volume = df["volume"] if "volume" in df.columns else pd.Series(0, index=df.index)

    # ── Moving Averages ──────────────────────────────────────────────────────
    df[f"sma_{SMA_SHORT}"]  = close.rolling(SMA_SHORT).mean()
    df[f"sma_{SMA_LONG}"]   = close.rolling(SMA_LONG).mean()
    df[f"ema_{EMA_SHORT}"]  = close.ewm(span=EMA_SHORT, adjust=False).mean()

    # ── RSI ─────────────────────────────────────────────────────────────────
    df[f"rsi_{RSI_PERIOD}"] = _rsi(close, RSI_PERIOD)
    df[f"rsi_{RSI_SHORT}"]  = _rsi(close, RSI_SHORT)

    # ── MACD ─────────────────────────────────────────────────────────────────
    ema_fast = close.ewm(span=MACD_FAST,  adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW,  adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_line"]     = macd_line
    df["macd_signal"]   = signal_line
    df["macd_histogram"]= macd_line - signal_line

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid = close.rolling(BB_PERIOD).mean()
    bb_std = close.rolling(BB_PERIOD).std()
    df["bb_upper"]  = bb_mid + BB_STD * bb_std
    df["bb_middle"] = bb_mid
    df["bb_lower"]  = bb_mid - BB_STD * bb_std
    df["bb_pct"]    = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    # ── ATR ───────────────────────────────────────────────────────────────────
    df[f"atr_{ATR_PERIOD}"] = _atr(high, low, close, ATR_PERIOD)

    # ── OBV ───────────────────────────────────────────────────────────────────
    df["obv"] = _obv(close, volume)

    # ── Volume SMA ────────────────────────────────────────────────────────────
    df[f"volume_sma_{VOL_SMA}"] = volume.rolling(VOL_SMA).mean()
    df["breakout_volume"] = volume > (BREAKOUT_VOLUME_MULT * df[f"volume_sma_{VOL_SMA}"])

    # ── Golden / Death Cross ──────────────────────────────────────────────────
    sma_s = df[f"sma_{SMA_SHORT}"]
    sma_l = df[f"sma_{SMA_LONG}"]
    df["golden_cross"] = (sma_s > sma_l) & (sma_s.shift(1) <= sma_l.shift(1))
    df["death_cross"]  = (sma_s < sma_l) & (sma_s.shift(1) >= sma_l.shift(1))
    # Flag: has there been a golden/death cross in last 5 days?
    df["recent_golden_cross"] = df["golden_cross"].rolling(5).max().astype(bool)
    df["recent_death_cross"]  = df["death_cross"].rolling(5).max().astype(bool)

    # ── RSI Divergence ────────────────────────────────────────────────────────
    # Bullish divergence: price makes lower low, RSI makes higher low (last DIVERGENCE_WINDOW bars)
    rsi = df[f"rsi_{RSI_PERIOD}"]
    df["rsi_bullish_div"]  = _bullish_divergence(close, rsi, DIVERGENCE_WINDOW)
    df["rsi_bearish_div"]  = _bearish_divergence(close, rsi, DIVERGENCE_WINDOW)

    # ── MACD Divergence ───────────────────────────────────────────────────────
    hist = df["macd_histogram"]
    df["macd_bullish_div"] = _bullish_divergence(close, hist, DIVERGENCE_WINDOW)
    df["macd_bearish_div"] = _bearish_divergence(close, hist, DIVERGENCE_WINDOW)

    # ── OBV Trend (divergence: OBV trending opposite to price) ───────────────
    price_slope = close.diff(5)
    obv_slope   = df["obv"].diff(5)
    df["obv_bullish_div"] = (price_slope < 0) & (obv_slope > 0)
    df["obv_bearish_div"] = (price_slope > 0) & (obv_slope < 0)

    # ── Above/Below Key MAs ───────────────────────────────────────────────────
    df["above_sma200"] = close > sma_l
    df["above_sma50"]  = close > sma_s

    return df


def get_latest_signals(df: pd.DataFrame) -> dict:
    """Return the most recent row's indicator values as a dict."""
    if df.empty:
        return {}
    last = df.iloc[-1]
    return {
        "close":           last.get("close"),
        f"sma_{SMA_SHORT}": last.get(f"sma_{SMA_SHORT}"),
        f"sma_{SMA_LONG}":  last.get(f"sma_{SMA_LONG}"),
        f"rsi_{RSI_PERIOD}": last.get(f"rsi_{RSI_PERIOD}"),
        f"rsi_{RSI_SHORT}": last.get(f"rsi_{RSI_SHORT}"),
        "macd_line":        last.get("macd_line"),
        "macd_signal":      last.get("macd_signal"),
        "macd_histogram":   last.get("macd_histogram"),
        "bb_pct":           last.get("bb_pct"),
        "bb_lower":         last.get("bb_lower"),
        f"atr_{ATR_PERIOD}": last.get(f"atr_{ATR_PERIOD}"),
        "obv":              last.get("obv"),
        "above_sma200":     last.get("above_sma200"),
        "above_sma50":      last.get("above_sma50"),
        "recent_golden_cross": last.get("recent_golden_cross"),
        "recent_death_cross":  last.get("recent_death_cross"),
        "rsi_bullish_div":  last.get("rsi_bullish_div"),
        "rsi_bearish_div":  last.get("rsi_bearish_div"),
        "macd_bullish_div": last.get("macd_bullish_div"),
        "obv_bullish_div":  last.get("obv_bullish_div"),
        "breakout_volume":  last.get("breakout_volume"),
    }


def compute_and_cache(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators and save as parquet cache."""
    result = compute_indicators(df)
    cache_path = CACHE_DIR / f"{ticker}.parquet"
    try:
        result.to_parquet(cache_path, index=False)
    except Exception as e:
        log.debug("Cache write failed for %s: %s", ticker, e)
    return result


def load_from_cache(ticker: str) -> Optional[pd.DataFrame]:
    """Load cached indicator DataFrame if it exists."""
    cache_path = CACHE_DIR / f"{ticker}.parquet"
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            log.debug("Cache read failed for %s: %s", ticker, e)
    return None


def get_indicators_for_ticker(ticker: str) -> Optional[pd.DataFrame]:
    """
    Get indicators from cache if fresh (today), else recompute from DB prices.
    """
    from db.schema import get_connection
    from db.queries import get_stock_id, get_prices

    cached = load_from_cache(ticker)
    if cached is not None and not cached.empty:
        last_date = pd.to_datetime(cached["date"]).max()
        if last_date.date() >= (pd.Timestamp.today() - pd.Timedelta(days=1)).date():
            return cached

    conn = get_connection()
    sid = get_stock_id(conn, ticker)
    if sid is None:
        conn.close()
        return None
    df = get_prices(conn, sid)
    conn.close()

    if df.empty:
        return None
    return compute_and_cache(ticker, df)


# ─── Private indicator helpers ────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range."""
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low  - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _bullish_divergence(price: pd.Series, indicator: pd.Series, window: int) -> pd.Series:
    """
    Detect bullish divergence: price lower low + indicator higher low.
    Returns boolean Series.
    """
    price_ll = price < price.rolling(window).min().shift(1)
    ind_hl   = indicator > indicator.rolling(window).min().shift(1)
    return price_ll & ind_hl


def _bearish_divergence(price: pd.Series, indicator: pd.Series, window: int) -> pd.Series:
    """
    Detect bearish divergence: price higher high + indicator lower high.
    """
    price_hh = price > price.rolling(window).max().shift(1)
    ind_lh   = indicator < indicator.rolling(window).max().shift(1)
    return price_hh & ind_lh


# ─── Batch update ─────────────────────────────────────────────────────────────

def update_technicals_cache() -> None:
    """Recompute and cache technical indicators for all active tickers."""
    from db.schema import get_connection
    from db.queries import get_active_universe, get_stock_id, get_prices

    conn = get_connection()
    universe = get_active_universe(conn)

    for _, row in universe.iterrows():
        ticker = row["ticker"]
        sid = get_stock_id(conn, ticker)
        if sid is None:
            continue
        df = get_prices(conn, sid)
        if df.empty:
            continue
        try:
            compute_and_cache(ticker, df)
        except Exception as e:
            log.warning("Technicals compute failed for %s: %s", ticker, e)

    conn.close()
    log.info("Technicals cache updated for %d tickers", len(universe))
