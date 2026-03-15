"""
Horizon Ledger — High-Risk Turnaround Scoring
Identifies distressed companies showing early signs of fundamental improvement.

Hard exclusions:
  - Beneish M-Score > -1.78 (fraud signal)
  - 3+ consecutive Z-Score declines
  - Revenue declining >30% YoY with no catalyst

Position size caps enforced in index builder: 3–5% max, 20% total.
Time stop: flag for review after 2 quarters without improvement.

Target: top 15 stocks (higher conviction, smaller set).
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    TURNAROUND_WEIGHTS,
    BENEISH_THRESHOLD,
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
    get_price_on_date,
)
from scoring.composite import (
    percentile_rank,
    weighted_composite_score,
    altman_z_from_row,
    beneish_m_score,
    beneish_screen_score,
    safe_div,
    MISSING_FACTOR_FILL as _FILL,
)

log = logging.getLogger(__name__)

STRATEGY = "turnaround"


def compute_raw_factors(
    ticker: str,
    stock_id: int,
    conn,
    as_of: str,
    market_cap: Optional[float] = None,
    sector: Optional[str] = None,
) -> tuple[dict, list[str]]:
    """Returns (factors_dict, exclusion_reasons)."""
    factors: dict[str, Optional[float]] = {}
    exclusions: list[str] = []

    fdf = get_fundamentals_as_of(conn, stock_id, as_of, n_quarters=12)
    if fdf.empty or len(fdf) < 2:
        exclusions.append("Insufficient fundamental data")
        return factors, exclusions

    curr = fdf.iloc[0]
    q1   = fdf.iloc[1] if len(fdf) > 1 else None
    q2   = fdf.iloc[2] if len(fdf) > 2 else None
    q3   = fdf.iloc[3] if len(fdf) > 3 else None
    q4   = fdf.iloc[4] if len(fdf) > 4 else None

    # ── Hard Exclusion 1: Beneish M-Score > -1.78 ────────────────────────────
    m_score = None
    if q1 is not None:
        m_score = beneish_m_score(curr, q1)
    if m_score is not None and m_score > BENEISH_THRESHOLD:
        exclusions.append(f"Beneish M={m_score:.2f} > {BENEISH_THRESHOLD} (manipulation risk)")
        return factors, exclusions

    # ── Hard Exclusion 2: 3+ consecutive Z-Score declines ────────────────────
    z_scores = []
    for q in [curr, q1, q2, q3]:
        if q is not None:
            z = altman_z_from_row(q, market_cap=market_cap)
            if z is not None:
                z_scores.append(z)

    if len(z_scores) >= 3:
        # Check if Z-Score declining for last 3 periods
        if all(z_scores[i] < z_scores[i+1] for i in range(len(z_scores)-1)):
            exclusions.append("3+ consecutive Z-Score declines")
            return factors, exclusions

    # ── Hard Exclusion 3: Revenue declining >30% with no catalyst ────────────
    rev_curr = curr.get("revenue")
    rev_4q   = q4.get("revenue") if q4 is not None else None
    if rev_curr and rev_4q and rev_4q > 0:
        yoy_rev_change = (rev_curr - rev_4q) / rev_4q
        if yoy_rev_change < -0.30:
            exclusions.append(f"Revenue declined {yoy_rev_change:.0%} YoY")
            return factors, exclusions

    # ── Beneish M-Score Factor (binary tiers) ────────────────────────────────
    factors["beneish_m_screen"] = beneish_screen_score(m_score)

    # ── Z-Score Level ─────────────────────────────────────────────────────────
    z_curr = altman_z_from_row(curr, market_cap=market_cap) if curr is not None else None
    # Prefer Grey zone that's improving over safe zone
    # Scale: distress zone (0–1.81) → 0–50, grey zone (1.81–2.99) → 50–80, safe → 80–100
    if z_curr is not None:
        if z_curr < 0:
            factors["z_score_level"] = 10.0
        elif z_curr < 1.81:
            factors["z_score_level"] = z_curr / 1.81 * 40
        elif z_curr < 2.99:
            factors["z_score_level"] = 40 + (z_curr - 1.81) / (2.99 - 1.81) * 40
        else:
            factors["z_score_level"] = 80.0 + min(20, (z_curr - 2.99) * 5)
    else:
        factors["z_score_level"] = None

    # ── Z-Score Trajectory (slope over last 2 quarters) ──────────────────────
    if len(z_scores) >= 2:
        slope = z_scores[0] - z_scores[1]   # Positive slope = improving
        factors["z_score_trajectory"] = slope
    else:
        factors["z_score_trajectory"] = None

    # ── Revenue Trajectory ────────────────────────────────────────────────────
    rev_series = [fdf.iloc[i].get("revenue") for i in range(min(6, len(fdf)))]
    factors["revenue_trajectory"] = _compute_slope(rev_series)

    # ── Operating Margin Trajectory ───────────────────────────────────────────
    margin_series = []
    for i in range(min(6, len(fdf))):
        r = fdf.iloc[i].get("revenue")
        e = fdf.iloc[i].get("ebit")
        m = safe_div(e, r)
        margin_series.append(m)
    factors["margin_trajectory"] = _compute_slope([x for x in margin_series if x is not None])

    # ── FCF Trajectory ────────────────────────────────────────────────────────
    fcf_series = [fdf.iloc[i].get("free_cash_flow") for i in range(min(6, len(fdf)))]
    factors["fcf_trajectory"] = _compute_slope(fcf_series)

    # ── Insider Cluster (placeholder — no free real-time source) ─────────────
    # Set to neutral (50) until insider data source is integrated
    factors["insider_cluster"] = _FILL

    # ── Short Interest (not available free — set neutral) ─────────────────────
    factors["short_interest_inv"] = _FILL

    # ── RSI Divergence ────────────────────────────────────────────────────────
    try:
        from pipeline.technicals import get_indicators_for_ticker
        tech_df = get_indicators_for_ticker(ticker)
        if tech_df is not None and not tech_df.empty:
            last = tech_df.iloc[-1]
            factors["rsi_divergence"] = 100.0 if last.get("rsi_bullish_div") else 0.0
        else:
            factors["rsi_divergence"] = _FILL
    except Exception:
        factors["rsi_divergence"] = _FILL

    # ── Industry Cycle (manual classification — set neutral) ─────────────────
    # TODO: Map sectors to cycle phase (recovering/stable/declining)
    factors["industry_cycle"] = _FILL

    # ── Relative Value (P/B vs sector) ────────────────────────────────────────
    bvps = curr.get("book_value_per_share")
    shares = curr.get("shares_outstanding")
    price_per_share = safe_div(market_cap, shares) if market_cap and shares else None
    pb = safe_div(price_per_share, bvps)
    # For turnaround: low P/B (but not zero/negative book) is attractive
    if pb is not None and 0 < pb < 10:
        factors["relative_value"] = max(0, 100 - pb * 10)
    else:
        factors["relative_value"] = _FILL

    return factors, exclusions


def _compute_slope(values: list) -> Optional[float]:
    """Fit a simple linear slope to a series of values. Positive = improving."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    try:
        import numpy as np
        x = np.arange(len(vals), dtype=float)
        # Reverse so index 0 = most recent, we want slope of improvement over time
        y = np.array(list(reversed(vals)), dtype=float)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    except Exception:
        return None


def score_universe(as_of: Optional[str] = None, persist: bool = True) -> pd.DataFrame:
    """Score all active stocks on the turnaround strategy."""
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
            raw, excl = compute_raw_factors(tkr, sid, conn, as_of, market_cap=mcap, sector=sector)
            raw["ticker"]            = tkr
            raw["stock_id"]          = sid
            raw["sector"]            = sector
            raw["_excluded"]         = bool(excl)
            raw["_exclusion_reasons"]= "; ".join(excl) if excl else ""
            all_factors.append(raw)
        except Exception as e:
            log.debug("Turnaround factor error for %s: %s", tkr, e)
            all_factors.append({"ticker": tkr, "stock_id": sid, "sector": sector, "_excluded": True, "_exclusion_reasons": str(e)})

    df = pd.DataFrame(all_factors)
    if df.empty:
        conn.close()
        return df

    df_excl  = df[df["_excluded"]].copy()
    df_valid = df[~df["_excluded"]].copy()

    factor_cols = [f for f in weights if f in df_valid.columns]
    for col in factor_cols:
        df_valid[col + "_pct"] = percentile_rank(df_valid[col])

    # ── Step 2.5: Sector neutralization ──────────────────────────────────────
    if SECTOR_NEUTRALIZE and "sector" in df_valid.columns:
        from scoring.utils import sector_neutralize
        pct_cols = [f + "_pct" for f in factor_cols if f + "_pct" in df_valid.columns]
        if pct_cols:
            df_valid = sector_neutralize(df_valid, pct_cols, sector_col="sector")

    def _composite(row):
        factor_scores = {f: row.get(f + "_pct", _FILL) for f in weights}
        return weighted_composite_score(factor_scores, weights)

    if not df_valid.empty:
        df_valid["composite_score"] = df_valid.apply(_composite, axis=1)
        df_valid = df_valid.sort_values("composite_score", ascending=False)
        df_valid["rank"] = range(1, len(df_valid) + 1)

    df_excl["composite_score"] = 0.0
    df_excl["rank"] = None
    df_out = pd.concat([df_valid, df_excl], ignore_index=True)

    if persist and not df_valid.empty:
        with conn:
            for _, row in df_valid.iterrows():
                sid   = int(row["stock_id"])
                score = float(row["composite_score"]) if pd.notna(row.get("composite_score")) else 0.0
                comps = {f + "_pct": float(row.get(f + "_pct", _FILL)) for f in factor_cols}
                upsert_score(conn, sid, as_of, STRATEGY, score, comps, weight_version)
                rank  = int(row["rank"]) if pd.notna(row.get("rank")) else 9999
                upsert_prediction(conn, sid, STRATEGY, as_of, score, rank, comps)
            conn.commit()

    conn.close()
    log.info("Turnaround scoring complete. %d eligible, %d excluded.", len(df_valid), len(df_excl))
    return df_out
