"""
Horizon Ledger — Long-Term Buy-and-Hold Scoring
Composite factor model targeting quality + value + growth stocks.

Factor weights (initial, from config.py — updated via reweighting system):
  gross_profitability, roic, piotroski_f, earnings_yield, fcf_yield,
  ev_to_ebitda_inv, price_to_sales_inv, price_to_book_inv, altman_z,
  debt_to_equity_inv, current_ratio, revenue_cagr_5y, earnings_cagr_5y,
  momentum_6m

Risk overlay (post-ranking exclusions):
  - Exclude stocks below 200-day SMA
  - Exclude Altman Z < 1.81

Target: top 25 stocks. Rebalance semi-annually.

References:
  Novy-Marx (2013): gross profitability — predictive power
  Piotroski (2000): F-Score value investing signal
  Altman (1968): Z-Score distress detection
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    LONG_TERM_WEIGHTS,
    ALTMAN_DISTRESS,
    MISSING_FACTOR_FILL,
    INITIAL_WEIGHT_VERSION,
    SECTOR_NEUTRALIZE,
)
from db.schema import get_connection
from db.queries import (
    get_active_universe,
    get_latest_fundamentals,
    get_fundamentals_as_of,
    get_active_weights,
    get_latest_weight_version,
    upsert_score,
    upsert_prediction,
    get_stock_id,
)
from scoring.composite import (
    percentile_rank, weighted_composite_score,
    piotroski_f_score, piotroski_to_percentile,
    altman_z_from_row, compute_enterprise_value,
    compute_cagr, safe_div, MISSING_FACTOR_FILL as _FILL,
)

log = logging.getLogger(__name__)

STRATEGY = "long_term"


def compute_raw_factors(
    ticker: str,
    stock_id: int,
    conn,
    as_of: str,
    market_cap: Optional[float] = None,
) -> dict:
    """
    Compute all raw (un-normalized) factor values for a single stock.
    Returns dict of {factor_name: value}.  None = data unavailable.
    """
    factors: dict[str, Optional[float]] = {}

    # Get fundamentals (point-in-time)
    fdf = get_fundamentals_as_of(conn, stock_id, as_of, n_quarters=20)
    if fdf.empty:
        return factors

    curr = fdf.iloc[0]  # Most recent
    prev_4 = fdf.iloc[4] if len(fdf) > 4 else None   # ~1 year ago
    prev_20 = fdf.iloc[-1] if len(fdf) >= 20 else (fdf.iloc[-1] if len(fdf) > 1 else None)

    ta   = curr.get("total_assets")
    eq   = curr.get("total_equity")
    td   = curr.get("total_debt")
    rev  = curr.get("revenue")
    gp   = curr.get("gross_profit")
    ebit = curr.get("ebit")
    fcf  = curr.get("free_cash_flow")
    ni   = curr.get("net_income")
    ca   = curr.get("current_assets")
    cl   = curr.get("current_liabilities")
    cash = curr.get("cash")
    eps  = curr.get("eps")
    bvps = curr.get("book_value_per_share")
    shares = curr.get("shares_outstanding")

    # Enterprise Value
    ev = compute_enterprise_value(market_cap, td, cash) if market_cap else None

    # ── Profitability / Quality ────────────────────────────────────────────────
    # Gross Profitability = GP / Total Assets (Novy-Marx 2013)
    factors["gross_profitability"] = safe_div(gp, ta)

    # ROIC = EBIT / (Total Debt + Total Equity)
    invested_capital = (td or 0) + (eq or 0)
    factors["roic"] = safe_div(ebit, invested_capital if invested_capital != 0 else None)

    # Piotroski F-Score
    f = piotroski_f_score(curr, prev_4)
    factors["piotroski_f"] = float(f)   # raw 0–9 (percentile applied cross-sectionally)

    # ── Value ─────────────────────────────────────────────────────────────────
    # Earnings Yield = EBIT / EV
    factors["earnings_yield"] = safe_div(ebit, ev)

    # FCF Yield = FCF / EV
    factors["fcf_yield"] = safe_div(fcf, ev)

    # EV/EBITDA (inverted — lower multiple is better)
    ebitda_proxy = (ebit or 0)   # simplified: no depreciation in schema
    ev_ebitda = safe_div(ev, ebitda_proxy if ebitda_proxy != 0 else None)
    factors["ev_to_ebitda_inv"] = (1 / ev_ebitda) if (ev_ebitda and ev_ebitda > 0) else None

    # P/S (inverted)
    price_per_share = (market_cap / shares) if (market_cap and shares and shares > 0) else None
    rev_per_share = safe_div(rev, shares)
    ps = safe_div(price_per_share, rev_per_share)
    factors["price_to_sales_inv"] = (1 / ps) if (ps and ps > 0) else None

    # P/B (inverted)
    pb = safe_div(price_per_share, bvps)
    factors["price_to_book_inv"] = (1 / pb) if (pb and pb > 0) else None

    # ── Safety ────────────────────────────────────────────────────────────────
    # Altman Z-Score
    factors["altman_z"] = altman_z_from_row(curr, market_cap=market_cap)

    # Debt to Equity (inverted)
    de = safe_div(td, eq)
    factors["debt_to_equity_inv"] = (1 / de) if (de and de > 0) else (100.0 if de == 0 else None)

    # Current Ratio
    factors["current_ratio"] = safe_div(ca, cl)

    # ── Growth ────────────────────────────────────────────────────────────────
    # Revenue CAGR 5 years
    if len(fdf) >= 20 and prev_20 is not None:
        rev_now  = rev or 0
        rev_5y   = prev_20.get("revenue") or 0
        factors["revenue_cagr_5y"] = compute_cagr([rev_5y, rev_now], years=5) if rev_5y > 0 else None
    else:
        factors["revenue_cagr_5y"] = None

    # Earnings (EPS) CAGR 5 years
    if len(fdf) >= 20 and prev_20 is not None:
        eps_now  = eps or 0
        eps_5y   = prev_20.get("eps") or 0
        if eps_5y > 0 and eps_now > 0:
            factors["earnings_cagr_5y"] = compute_cagr([eps_5y, eps_now], years=5)
        else:
            factors["earnings_cagr_5y"] = None
    else:
        factors["earnings_cagr_5y"] = None

    # ── Momentum ──────────────────────────────────────────────────────────────
    # 6-month return, skipping most recent month (Jegadeesh & Titman 1993 refinement)
    from db.queries import get_price_on_date
    try:
        p_now = get_price_on_date(conn, stock_id, as_of)
        p_1m  = get_price_on_date(conn, stock_id, (date.fromisoformat(as_of) - timedelta(days=21)).isoformat())
        p_7m  = get_price_on_date(conn, stock_id, (date.fromisoformat(as_of) - timedelta(days=210)).isoformat())
        if p_1m and p_7m and p_7m > 0:
            factors["momentum_6m"] = (p_1m - p_7m) / p_7m
        else:
            factors["momentum_6m"] = None
    except Exception:
        factors["momentum_6m"] = None

    return factors


def score_universe(as_of: Optional[str] = None, persist: bool = True) -> pd.DataFrame:
    """
    Score all active stocks on the long-term strategy.
    Returns DataFrame with columns: ticker, composite_score, [factor columns].
    Persists scores and predictions to DB if persist=True.
    """
    if as_of is None:
        as_of = date.today().isoformat()

    conn = get_connection()
    universe = get_active_universe(conn)
    weights = get_active_weights(conn, STRATEGY)
    weight_version = get_latest_weight_version(conn, STRATEGY) or INITIAL_WEIGHT_VERSION

    log.info("Scoring %d stocks on '%s' strategy (as of %s)...", len(universe), STRATEGY, as_of)

    # ── Step 1: Collect raw factors for all stocks ────────────────────────────
    all_factors: list[dict] = []
    for _, row in universe.iterrows():
        tkr = row["ticker"]
        sid = row["id"]
        mcap = row.get("market_cap")
        try:
            raw = compute_raw_factors(tkr, sid, conn, as_of, market_cap=mcap)
            raw["ticker"] = tkr
            raw["stock_id"] = sid
            raw["sector"] = row.get("sector")
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

    # ── Step 2.5: Sector neutralization ──────────────────────────────────────
    if SECTOR_NEUTRALIZE and "sector" in df.columns:
        from scoring.utils import sector_neutralize
        pct_cols = [f + "_pct" for f in factor_cols if f + "_pct" in df.columns]
        if pct_cols:
            df = sector_neutralize(df, pct_cols, sector_col="sector")

    # ── Step 3: Compute weighted composite score ──────────────────────────────
    def _composite(row):
        factor_scores = {f: row.get(f + "_pct", _FILL) for f in weights}
        return weighted_composite_score(factor_scores, weights)

    df["composite_score"] = df.apply(_composite, axis=1)

    # ── Step 4: Risk overlay — mark exclusions ────────────────────────────────
    def _passes_risk(row):
        z = row.get("altman_z")
        if z is not None and z < ALTMAN_DISTRESS:
            return False, "Altman Z < 1.81"
        return True, ""

    df["passes_risk"], df["exclusion_reason"] = zip(*df.apply(_passes_risk, axis=1))

    # ── Step 5: Rank ──────────────────────────────────────────────────────────
    df_ranked = df[df["passes_risk"]].copy()
    df_ranked = df_ranked.sort_values("composite_score", ascending=False)
    df_ranked["rank"] = range(1, len(df_ranked) + 1)
    df_excluded = df[~df["passes_risk"]].copy()
    df_excluded["rank"] = None
    df_out = pd.concat([df_ranked, df_excluded], ignore_index=True)

    # ── Step 6: Persist ───────────────────────────────────────────────────────
    if persist:
        with conn:
            for _, row in df_out.iterrows():
                sid = int(row["stock_id"])
                score = float(row["composite_score"]) if pd.notna(row.get("composite_score")) else 0.0
                comps = {f + "_pct": float(row.get(f + "_pct", _FILL)) for f in factor_cols}
                upsert_score(conn, sid, as_of, STRATEGY, score, comps, weight_version)
                rank = int(row["rank"]) if pd.notna(row.get("rank")) else 9999
                upsert_prediction(conn, sid, STRATEGY, as_of, score, rank, comps)
            conn.commit()

    conn.close()
    log.info("Long-term scoring complete. Top score: %.1f (%s)", df_ranked["composite_score"].max() if not df_ranked.empty else 0, df_ranked.iloc[0]["ticker"] if not df_ranked.empty else "N/A")
    return df_out
