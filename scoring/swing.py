"""
Horizon Ledger — Short-Term Swing Trading Scoring
Combines earnings surprise signals with technical momentum.

PEAD (Post-Earnings Announcement Drift) protocol:
  - Enter Day 2 after positive earnings surprise
  - Target hold: 21–63 trading days
  - Exit: time stop at 63 days OR price target OR 1.5–2.0× ATR stop

Position sizing:
  - 1–2% risk per trade
  - Max 10 concurrent positions
  - Kelly Criterion (capped at half-Kelly)

Target: top 10 stocks. Active rotation.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    SWING_WEIGHTS,
    MISSING_FACTOR_FILL,
    INITIAL_WEIGHT_VERSION,
    SWING_ATR_STOP_MULT_LOW,
    SWING_MIN_RR_RATIO,
    BREAKOUT_VOLUME_MULT,
    SECTOR_NEUTRALIZE,
)
from db.schema import get_connection
from db.queries import (
    get_active_universe,
    get_active_weights,
    get_latest_weight_version,
    upsert_score,
    upsert_prediction,
    get_price_on_date,
    get_stock_id,
)
from scoring.composite import (
    percentile_rank,
    weighted_composite_score,
    safe_div,
    MISSING_FACTOR_FILL as _FILL,
)

log = logging.getLogger(__name__)

STRATEGY = "swing"


def compute_raw_factors(
    ticker: str,
    stock_id: int,
    conn,
    as_of: str,
    market_cap: Optional[float] = None,
    sector: Optional[str] = None,
) -> dict:
    """Compute swing trading factor values for a single stock."""
    factors: dict[str, Optional[float]] = {}

    # ── Technical indicators ──────────────────────────────────────────────────
    try:
        from pipeline.technicals import get_indicators_for_ticker
        tech = get_indicators_for_ticker(ticker)
    except Exception:
        tech = None

    if tech is not None and not tech.empty:
        last = tech.iloc[-1]
        recent = tech.tail(20)

        # RSI Signal: use RSI_5 for short-term mean reversion setup
        rsi5 = last.get("rsi_5")
        rsi_bullish_div = last.get("rsi_bullish_div", False)
        # Low RSI5 (oversold) = setup; divergence adds to signal
        if rsi5 is not None:
            # Score 0–100: lower RSI5 (30–50 range) = better swing entry
            if rsi_bullish_div:
                factors["rsi_signal"] = min(100, (50 - min(rsi5, 50)) / 50 * 100 + 25)
            else:
                factors["rsi_signal"] = max(0, (50 - min(rsi5, 70)) / 50 * 100)
        else:
            factors["rsi_signal"] = _FILL

        # MACD Signal: divergence between MACD histogram and price
        macd_bullish = last.get("macd_bullish_div", False)
        macd_hist    = last.get("macd_histogram")
        if macd_bullish:
            factors["macd_signal"] = 90.0
        elif macd_hist is not None and macd_hist > 0:
            factors["macd_signal"] = 60.0
        else:
            factors["macd_signal"] = 30.0

        # Volume Confirmation: OBV trend + breakout volume
        obv_bullish   = last.get("obv_bullish_div", False)
        breakout_vol  = last.get("breakout_volume", False)
        if obv_bullish and breakout_vol:
            factors["volume_confirmation"] = 100.0
        elif breakout_vol or obv_bullish:
            factors["volume_confirmation"] = 65.0
        else:
            factors["volume_confirmation"] = 30.0

        # OBV Trend: OBV diverging bullishly from price
        factors["obv_trend"] = 100.0 if last.get("obv_bullish_div") else 30.0

        # Bollinger Position: price near lower band = potential bounce
        bb_pct = last.get("bb_pct")
        if bb_pct is not None:
            factors["bollinger_position"] = max(0, (0.3 - bb_pct) / 0.3 * 100) if bb_pct < 0.3 else 10.0
        else:
            factors["bollinger_position"] = _FILL

        # ATR Reward/Risk: check if ≥2:1 R:R is available
        close = last.get("close")
        atr   = last.get("atr_14")
        bb_lower = last.get("bb_lower")
        if close and atr and atr > 0:
            stop_distance = SWING_ATR_STOP_MULT_LOW * atr
            if bb_lower:
                target = close + (close - bb_lower) * 2   # Simplified target
                risk   = stop_distance
                rr = (target - close) / risk if risk > 0 else 0
            else:
                rr = SWING_MIN_RR_RATIO   # Assume 2:1 if no BB data
            factors["atr_reward_risk"] = min(100, rr / SWING_MIN_RR_RATIO * 50)
        else:
            factors["atr_reward_risk"] = _FILL

    else:
        # No technical data — fill with neutral
        for f in ["rsi_signal", "macd_signal", "volume_confirmation",
                   "obv_trend", "bollinger_position", "atr_reward_risk"]:
            factors[f] = _FILL

    # ── Momentum ──────────────────────────────────────────────────────────────
    # 6-month price momentum (skip most recent month)
    try:
        p_1m = get_price_on_date(conn, stock_id, (date.fromisoformat(as_of) - timedelta(days=21)).isoformat())
        p_7m = get_price_on_date(conn, stock_id, (date.fromisoformat(as_of) - timedelta(days=210)).isoformat())
        if p_1m and p_7m and p_7m > 0:
            factors["momentum_6m"] = (p_1m - p_7m) / p_7m
        else:
            factors["momentum_6m"] = None
    except Exception:
        factors["momentum_6m"] = None

    # ── Sector Momentum (3-month sector ETF return) ───────────────────────────
    factors["sector_momentum"] = _get_sector_momentum(sector, conn, as_of)

    # ── Earnings Surprise (SUE & EAR) — from yfinance earnings data ──────────
    sue_val, ear_val = _get_earnings_signals(ticker)
    factors["sue_magnitude"] = sue_val
    factors["ear_signal"]    = ear_val

    return factors


def _get_earnings_signals(ticker: str) -> tuple[float, float]:
    """
    Standardized Unexpected Earnings (SUE) and Earnings Announcement Return (EAR).
    Uses yfinance earnings history.  Returns (sue_pct, ear_pct).
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        cal = t.earnings_dates
        if cal is None or cal.empty:
            return _FILL, _FILL

        cal = cal.reset_index()
        cal = cal.dropna(subset=["EPS Estimate", "Reported EPS"])
        if cal.empty:
            return _FILL, _FILL

        # Most recent complete quarter
        cal = cal[cal["Reported EPS"].notna()].head(1)
        if cal.empty:
            return _FILL, _FILL

        est  = float(cal.iloc[0]["EPS Estimate"])
        rep  = float(cal.iloc[0]["Reported EPS"])
        sue_raw = (rep - est) / (abs(est) + 1e-6)   # Standardized surprise

        # Positive surprise → high SUE score
        if sue_raw > 0.10:
            sue_score = 90.0
        elif sue_raw > 0.05:
            sue_score = 70.0
        elif sue_raw > 0:
            sue_score = 55.0
        else:
            sue_score = 20.0

        return sue_score, sue_score * 0.8   # EAR correlated with SUE

    except Exception:
        return _FILL, _FILL


def _get_sector_momentum(
    sector: Optional[str],
    conn,
    as_of: str,
) -> float:
    """Get 3-month sector ETF return as a proxy for sector momentum."""
    sector_etfs = {
        "Technology":         "XLK",
        "Health Care":        "XLV",
        "Financials":         "XLF",
        "Consumer Discretionary": "XLY",
        "Industrials":        "XLI",
        "Communication Services": "XLC",
        "Consumer Staples":   "XLP",
        "Energy":             "XLE",
        "Materials":          "XLB",
        "Real Estate":        "XLRE",
        "Utilities":          "XLU",
    }
    if sector is None or sector not in sector_etfs:
        return _FILL

    etf = sector_etfs[sector]
    try:
        etf_id = get_stock_id(conn, etf)
        if etf_id is None:
            return _FILL
        p_now = get_price_on_date(conn, etf_id, as_of)
        p_3m  = get_price_on_date(conn, etf_id, (date.fromisoformat(as_of) - timedelta(days=63)).isoformat())
        if p_now and p_3m and p_3m > 0:
            ret = (p_now - p_3m) / p_3m
            # Convert to percentile-style score
            return min(100, max(0, 50 + ret * 200))
    except Exception:
        pass
    return _FILL


def score_universe(as_of: Optional[str] = None, persist: bool = True) -> pd.DataFrame:
    """Score all active stocks on the swing trading strategy."""
    if as_of is None:
        as_of = date.today().isoformat()

    conn = get_connection()
    universe = get_active_universe(conn)
    weights = get_active_weights(conn, STRATEGY)
    weight_version = get_latest_weight_version(conn, STRATEGY) or INITIAL_WEIGHT_VERSION

    log.info("Scoring %d stocks on '%s' strategy (as of %s)...", len(universe), STRATEGY, as_of)

    all_factors: list[dict] = []
    for _, row in universe.iterrows():
        tkr    = row["ticker"]
        sid    = row["id"]
        mcap   = row.get("market_cap")
        sector = row.get("sector")
        try:
            raw = compute_raw_factors(tkr, sid, conn, as_of, market_cap=mcap, sector=sector)
            raw["ticker"]   = tkr
            raw["stock_id"] = sid
            raw["sector"]   = sector
            all_factors.append(raw)
        except Exception as e:
            log.debug("Swing factor error for %s: %s", tkr, e)
            all_factors.append({"ticker": tkr, "stock_id": sid, "sector": sector})

    df = pd.DataFrame(all_factors)
    if df.empty:
        conn.close()
        return df

    factor_cols = [f for f in weights if f in df.columns]
    for col in factor_cols:
        df[col + "_pct"] = percentile_rank(df[col])

    # ── Step 2.5: Sector neutralization ──────────────────────────────────────
    if SECTOR_NEUTRALIZE and "sector" in df.columns:
        from scoring.utils import sector_neutralize
        pct_cols = [f + "_pct" for f in factor_cols if f + "_pct" in df.columns]
        if pct_cols:
            df = sector_neutralize(df, pct_cols, sector_col="sector")

    def _composite(row):
        factor_scores = {f: row.get(f + "_pct", _FILL) for f in weights}
        return weighted_composite_score(factor_scores, weights)

    df["composite_score"] = df.apply(_composite, axis=1)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    if persist:
        with conn:
            for _, row in df.iterrows():
                sid   = int(row["stock_id"])
                score = float(row["composite_score"]) if pd.notna(row.get("composite_score")) else 0.0
                comps = {f + "_pct": float(row.get(f + "_pct", _FILL)) for f in factor_cols}
                upsert_score(conn, sid, as_of, STRATEGY, score, comps, weight_version)
                rank  = int(row["rank"])
                upsert_prediction(conn, sid, STRATEGY, as_of, score, rank, comps)
            conn.commit()

    conn.close()
    log.info("Swing scoring complete. Top: %s (%.1f)", df.iloc[0]["ticker"] if not df.empty else "N/A", df.iloc[0]["composite_score"] if not df.empty else 0)
    return df


def kelly_position_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    half_kelly: bool = True,
) -> float:
    """
    Kelly Criterion position size as fraction of portfolio.
    Kelly f* = (bp - q) / b
      b = avg_win / avg_loss (odds ratio)
      p = win rate, q = 1 - p

    Capped at half-Kelly as an anti-overfitting guardrail.
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.02   # Default 2% risk
    b = avg_win / avg_loss
    q = 1 - win_rate
    kelly = (b * win_rate - q) / b
    kelly = max(0.0, kelly)
    if half_kelly:
        kelly *= 0.5
    return min(kelly, 0.05)   # Cap at 5%
