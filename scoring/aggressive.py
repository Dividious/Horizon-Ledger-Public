"""
Horizon Ledger — Aggressive Index Scoring
Aggressive Index — Momentum + Growth + Quality.
Target audience: young/long-horizon investors.

Factor weights (initial, from config.py — updated via reweighting system):
  momentum_12m, momentum_6m, sector_momentum,
  revenue_cagr_5y, earnings_cagr_5y, revenue_trajectory,
  gross_profitability, roic, piotroski_f,
  rsi_signal, volume_confirmation

Risk overlay (post-ranking exclusions):
  - Exclude Altman Z < 1.81 (ALTMAN_DISTRESS)
  - Exclude revenue YoY decline > 20%

Target: top 25 stocks. Rebalance monthly.

References:
  Jegadeesh & Titman (1993): Price momentum effect
  Fama & French (1996): 12-minus-1 month momentum standard
  Novy-Marx (2013): Gross profitability factor
  Piotroski (2000): F-Score quality screen
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    AGGRESSIVE_WEIGHTS,
    ALTMAN_DISTRESS,
    MISSING_FACTOR_FILL,
    INITIAL_WEIGHT_VERSION,
    SECTOR_NEUTRALIZE,
)
from db.schema import get_connection
from db.queries import (
    get_active_universe,
    get_fundamentals_as_of,
    get_active_weights,
    get_latest_weight_version,
    upsert_score,
    upsert_prediction,
    get_prices,
    get_price_on_date,
)
from scoring.composite import (
    percentile_rank, weighted_composite_score,
    piotroski_f_score,
    altman_z_from_row, compute_enterprise_value,
    compute_cagr, safe_div, MISSING_FACTOR_FILL as _FILL,
)
from scoring.utils import sector_neutralize

log = logging.getLogger(__name__)

STRATEGY = "aggressive"


def _compute_rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
    """
    Compute the most recent RSI value using Wilder's smoothing (EWM).
    Returns the last RSI value, or None if insufficient data.
    """
    if len(prices) < period + 1:
        return None
    try:
        delta = prices.diff()
        gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs    = gain / loss.replace(0, 1e-10)
        rsi   = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
    except Exception:
        return None


def compute_raw_factors(
    ticker: str,
    stock_id: int,
    conn,
    as_of: str,
    market_cap: Optional[float] = None,
    sector_medians: Optional[dict] = None,
) -> dict:
    """
    Compute all raw (un-normalized) factor values for a single stock.
    Returns dict of {factor_name: value}.  None = data unavailable.

    Args:
        sector_medians: Pre-computed {sector: median_6m_momentum} dict.
                        If None, sector_momentum will be None.
    """
    factors: dict[str, Optional[float]] = {}

    # Get fundamentals (point-in-time, up to 20 quarters for 5-year history)
    fdf = get_fundamentals_as_of(conn, stock_id, as_of, n_quarters=20)
    if fdf.empty:
        return factors

    curr    = fdf.iloc[0]                                                    # Most recent quarter
    prev_4  = fdf.iloc[4]  if len(fdf) > 4  else None                       # ~1 year ago
    prev_20 = fdf.iloc[-1] if len(fdf) >= 20 else (fdf.iloc[-1] if len(fdf) > 1 else None)

    ta  = curr.get("total_assets")
    eq  = curr.get("total_equity")
    td  = curr.get("total_debt")
    rev = curr.get("revenue")
    gp  = curr.get("gross_profit")
    ebit = curr.get("ebit")
    eps  = curr.get("eps")

    # ── Momentum ──────────────────────────────────────────────────────────────
    # Standard price lookback dates
    as_of_dt = date.fromisoformat(as_of)
    d_1m  = (as_of_dt - timedelta(days=21)).isoformat()    # ~1 month ago (skip month)
    d_7m  = (as_of_dt - timedelta(days=210)).isoformat()   # ~7 months ago
    d_13m = (as_of_dt - timedelta(days=390)).isoformat()   # ~13 months ago

    try:
        p_1m  = get_price_on_date(conn, stock_id, d_1m)
        p_7m  = get_price_on_date(conn, stock_id, d_7m)
        p_13m = get_price_on_date(conn, stock_id, d_13m)

        # 12-minus-1 month momentum (Fama-French standard)
        # Uses price 1 month ago vs 13 months ago to skip the reversal month
        if p_1m and p_13m and p_13m > 0:
            factors["momentum_12m"] = (p_1m - p_13m) / p_13m
        else:
            factors["momentum_12m"] = None

        # 6-month momentum (same skip-one-month convention)
        if p_1m and p_7m and p_7m > 0:
            factors["momentum_6m"] = (p_1m - p_7m) / p_7m
        else:
            factors["momentum_6m"] = None
    except Exception:
        factors["momentum_12m"] = None
        factors["momentum_6m"]  = None

    # Sector momentum: median 6m momentum of stocks in the same sector
    # Pre-computed in score_universe() and passed in via sector_medians
    try:
        sector = curr.get("sector") if hasattr(curr, "get") else None
        if sector_medians is not None and sector and sector in sector_medians:
            factors["sector_momentum"] = sector_medians[sector]
        else:
            factors["sector_momentum"] = None
    except Exception:
        factors["sector_momentum"] = None

    # ── Growth ────────────────────────────────────────────────────────────────
    # Revenue CAGR 5 years (20 quarters)
    if len(fdf) >= 20 and prev_20 is not None:
        rev_now = rev or 0
        rev_5y  = prev_20.get("revenue") or 0
        factors["revenue_cagr_5y"] = compute_cagr([rev_5y, rev_now], years=5) if rev_5y > 0 else None
    else:
        factors["revenue_cagr_5y"] = None

    # Earnings (EPS) CAGR 5 years
    if len(fdf) >= 20 and prev_20 is not None:
        eps_now = eps or 0
        eps_5y  = prev_20.get("eps") or 0
        if eps_5y > 0 and eps_now > 0:
            factors["earnings_cagr_5y"] = compute_cagr([eps_5y, eps_now], years=5)
        else:
            factors["earnings_cagr_5y"] = None
    else:
        factors["earnings_cagr_5y"] = None

    # Revenue trajectory: YoY quarterly revenue change = (rev_now - rev_4q_ago) / |rev_4q_ago|
    # Also stored as raw value for the risk overlay exclusion check.
    try:
        rev_4q = prev_4.get("revenue") if prev_4 is not None else None
        if rev is not None and rev_4q is not None and rev_4q != 0:
            factors["revenue_trajectory"] = (rev - rev_4q) / abs(rev_4q)
        else:
            factors["revenue_trajectory"] = None
        factors["_revenue_trajectory_raw"] = factors["revenue_trajectory"]   # For risk overlay
    except Exception:
        factors["revenue_trajectory"] = None
        factors["_revenue_trajectory_raw"] = None

    # ── Quality Filter ────────────────────────────────────────────────────────
    # Gross Profitability = GP / Total Assets (Novy-Marx 2013)
    factors["gross_profitability"] = safe_div(gp, ta)

    # ROIC = EBIT / (Total Debt + Total Equity)
    invested_capital = (td or 0) + (eq or 0)
    factors["roic"] = safe_div(ebit, invested_capital if invested_capital != 0 else None)

    # Piotroski F-Score (raw 0–9; percentile applied cross-sectionally)
    factors["piotroski_f"] = float(piotroski_f_score(curr, prev_4))

    # Also compute Altman Z for the risk overlay (not a scored factor here)
    factors["_altman_z"] = altman_z_from_row(curr, market_cap=market_cap)

    # ── Technical Confirmation ────────────────────────────────────────────────
    # RSI signal: stocks near RSI ~45 score highest (not overbought, not deeply oversold)
    # Formula: score = max(0, min(100, 100 - max(0, rsi - 45) * 2))
    # rsi=45→100, rsi=55→90, rsi=60→80, rsi=70→60, rsi=80→40, rsi=30→100 (but floor at 0)
    try:
        start_date = (as_of_dt - timedelta(days=90)).isoformat()
        price_df = get_prices(conn, stock_id, start=start_date, end=as_of)
        if price_df is not None and len(price_df) >= 20:
            prices = price_df["adj_close"].dropna()
            rsi_val = _compute_rsi(prices, period=14)
            if rsi_val is not None:
                rsi_score = max(0.0, min(100.0, 100.0 - max(0.0, rsi_val - 45.0) * 2.0))
                factors["rsi_signal"] = rsi_score
            else:
                factors["rsi_signal"] = None
        else:
            factors["rsi_signal"] = None
    except Exception:
        factors["rsi_signal"] = None

    # Volume confirmation: 20-day avg volume / 50-day avg volume ratio
    # Ratio > 1.2 → strong institutional participation; ratio < 0.8 → weak.
    # Score = min(100, (ratio / 1.5) * 100) — capped at 100.
    try:
        start_vol = (as_of_dt - timedelta(days=120)).isoformat()
        vol_df = get_prices(conn, stock_id, start=start_vol, end=as_of)
        if vol_df is not None and len(vol_df) >= 50:
            volumes = vol_df["volume"].dropna()
            avg_20d = float(volumes.iloc[-20:].mean()) if len(volumes) >= 20 else None
            avg_50d = float(volumes.iloc[-50:].mean()) if len(volumes) >= 50 else None
            if avg_20d is not None and avg_50d is not None and avg_50d > 0:
                ratio = avg_20d / avg_50d
                factors["volume_confirmation"] = min(100.0, (ratio / 1.5) * 100.0)
            else:
                factors["volume_confirmation"] = None
        else:
            factors["volume_confirmation"] = None
    except Exception:
        factors["volume_confirmation"] = None

    return factors


def score_universe(as_of: Optional[str] = None, persist: bool = True) -> pd.DataFrame:
    """
    Score all active stocks on the aggressive strategy.
    Returns DataFrame with columns: ticker, composite_score, [factor columns].
    Persists scores and predictions to DB if persist=True.
    """
    if as_of is None:
        as_of = date.today().isoformat()

    conn = get_connection()
    universe = get_active_universe(conn)
    weights = get_active_weights(conn, STRATEGY)
    weight_version = get_latest_weight_version(conn, STRATEGY) or INITIAL_WEIGHT_VERSION

    log.info(
        "Scoring %d stocks on '%s' strategy (as of %s)...",
        len(universe), STRATEGY, as_of,
    )

    # ── Pre-compute sector momentum medians ───────────────────────────────────
    # We need universe-level sector medians for the sector_momentum factor.
    # Step 1: collect each stock's 6m momentum, then group by sector for median.
    as_of_dt = date.fromisoformat(as_of)
    d_1m  = (as_of_dt - timedelta(days=21)).isoformat()
    d_7m  = (as_of_dt - timedelta(days=210)).isoformat()

    mom6m_by_sector: dict[str, list[float]] = {}
    for _, urow in universe.iterrows():
        try:
            sid    = urow["id"]
            sector = urow.get("sector")
            if not sector:
                continue
            p_1m = get_price_on_date(conn, sid, d_1m)
            p_7m = get_price_on_date(conn, sid, d_7m)
            if p_1m and p_7m and p_7m > 0:
                mom = (p_1m - p_7m) / p_7m
                mom6m_by_sector.setdefault(sector, []).append(mom)
        except Exception:
            continue

    sector_medians: dict[str, float] = {
        sector: float(np.median(vals))
        for sector, vals in mom6m_by_sector.items()
        if vals
    }
    log.debug("Sector momentum medians computed for %d sectors.", len(sector_medians))

    # ── Step 1: Collect raw factors for all stocks ────────────────────────────
    all_factors: list[dict] = []
    for _, row in universe.iterrows():
        tkr  = row["ticker"]
        sid  = row["id"]
        mcap = row.get("market_cap")
        try:
            raw = compute_raw_factors(
                tkr, sid, conn, as_of,
                market_cap=mcap,
                sector_medians=sector_medians,
            )
            raw["ticker"]   = tkr
            raw["stock_id"] = sid
            raw["sector"]   = row.get("sector")
            all_factors.append(raw)
        except Exception as e:
            log.debug("Raw factor error for %s: %s", tkr, e)
            all_factors.append({"ticker": tkr, "stock_id": sid, "sector": row.get("sector")})

    df = pd.DataFrame(all_factors)
    if df.empty:
        conn.close()
        return df

    # ── Step 2: Cross-sectional percentile rank each factor ──────────────────
    factor_cols = [f for f in weights if f in df.columns]
    for col in factor_cols:
        df[col + "_pct"] = percentile_rank(df[col])

    # Optional: sector-neutralize raw factor values before final percentile ranking
    if SECTOR_NEUTRALIZE and "sector" in df.columns:
        df = sector_neutralize(df, factor_cols=factor_cols, sector_col="sector")
        # Re-rank after neutralization
        for col in factor_cols:
            df[col + "_pct"] = percentile_rank(df[col])

    # ── Step 3: Compute weighted composite score ──────────────────────────────
    def _composite(row):
        factor_scores = {f: row.get(f + "_pct", _FILL) for f in weights}
        return weighted_composite_score(factor_scores, weights)

    df["composite_score"] = df.apply(_composite, axis=1)

    # ── Step 4: Risk overlay — mark exclusions ────────────────────────────────
    def _passes_risk(row):
        # Altman Z distress filter
        z = row.get("_altman_z")
        if z is not None and z < ALTMAN_DISTRESS:
            return False, "Altman Z < 1.81"

        # Revenue declining >20% YoY
        rev_traj = row.get("_revenue_trajectory_raw")
        if rev_traj is not None and rev_traj < -0.20:
            return False, "Revenue declining >20% YoY"

        return True, ""

    df["passes_risk"], df["exclusion_reason"] = zip(*df.apply(_passes_risk, axis=1))

    # ── Step 5: Rank ──────────────────────────────────────────────────────────
    df_ranked   = df[df["passes_risk"]].copy()
    df_ranked   = df_ranked.sort_values("composite_score", ascending=False)
    df_ranked["rank"] = range(1, len(df_ranked) + 1)
    df_excluded = df[~df["passes_risk"]].copy()
    df_excluded["rank"] = None
    df_out = pd.concat([df_ranked, df_excluded], ignore_index=True)

    # ── Step 6: Persist ───────────────────────────────────────────────────────
    if persist:
        with conn:
            for _, row in df_out.iterrows():
                sid   = int(row["stock_id"])
                score = float(row["composite_score"]) if pd.notna(row.get("composite_score")) else 0.0
                comps = {f + "_pct": float(row.get(f + "_pct", _FILL)) for f in factor_cols}
                upsert_score(conn, sid, as_of, STRATEGY, score, comps, weight_version)
                rank = int(row["rank"]) if pd.notna(row.get("rank")) else 9999
                upsert_prediction(conn, sid, STRATEGY, as_of, score, rank, comps)
            conn.commit()

    conn.close()
    log.info(
        "Aggressive scoring complete. Top score: %.1f (%s)",
        df_ranked["composite_score"].max() if not df_ranked.empty else 0,
        df_ranked.iloc[0]["ticker"] if not df_ranked.empty else "N/A",
    )
    return df_out
