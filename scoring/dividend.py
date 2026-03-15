"""
Horizon Ledger — Dividend / Income Scoring
Quality-first dividend model: safety → growth → yield → value.

Hard exclusion filters (applied BEFORE scoring):
  - Altman Z < 1.81
  - FCF payout ratio > 100%
  - Dividend yield > 2× sector median
  - No dividend increase in last 3 years

Chowder Rule (flagging, not in score):
  Yield + 5y_div_growth >= 12% (8% for utilities/REITs)

Target: top 25 stocks. Rebalance quarterly.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    DIVIDEND_WEIGHTS,
    ALTMAN_DISTRESS,
    CHOWDER_STANDARD,
    CHOWDER_UTILITY,
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
    get_stock_id,
    get_price_on_date,
)
from scoring.composite import (
    percentile_rank,
    weighted_composite_score,
    altman_z_from_row,
    compute_enterprise_value,
    compute_cagr,
    safe_div,
    MISSING_FACTOR_FILL as _FILL,
)

log = logging.getLogger(__name__)

STRATEGY = "dividend"
UTILITY_SECTORS = {"Utilities", "Real Estate"}


def compute_raw_factors(
    ticker: str,
    stock_id: int,
    conn,
    as_of: str,
    market_cap: Optional[float] = None,
    sector: Optional[str] = None,
) -> tuple[dict, list[str]]:
    """
    Returns (factors_dict, exclusion_reasons).
    exclusion_reasons is non-empty if stock should be hard-excluded.
    """
    factors: dict[str, Optional[float]] = {}
    exclusions: list[str] = []

    fdf = get_fundamentals_as_of(conn, stock_id, as_of, n_quarters=40)
    if fdf.empty:
        exclusions.append("No fundamental data")
        return factors, exclusions

    curr = fdf.iloc[0]

    ta   = curr.get("total_assets")
    eq   = curr.get("total_equity")
    td   = curr.get("total_debt")
    rev  = curr.get("revenue")
    ni   = curr.get("net_income")
    fcf  = curr.get("free_cash_flow")
    divs = curr.get("dividends_paid")   # Usually negative (cash outflow)
    ca   = curr.get("current_assets")
    cl   = curr.get("current_liabilities")
    cash = curr.get("cash")
    eps  = curr.get("eps")
    shares = curr.get("shares_outstanding")
    gp   = curr.get("gross_profit")
    ebit = curr.get("ebit")

    ev = compute_enterprise_value(market_cap, td, cash) if market_cap else None
    price_per_share = safe_div(market_cap, shares)

    # Dividends paid is typically negative in cash flow statements
    div_abs = abs(divs) if divs is not None else None
    div_per_share = safe_div(div_abs, shares)

    # ── Hard Exclusion 1: Altman Z < 1.81 ────────────────────────────────────
    z = altman_z_from_row(curr, market_cap=market_cap)
    if z is not None and z < ALTMAN_DISTRESS:
        exclusions.append(f"Altman Z={z:.2f} < {ALTMAN_DISTRESS}")

    # ── Hard Exclusion 2: FCF Payout > 100% ──────────────────────────────────
    fcf_payout = None
    if div_abs and fcf:
        fcf_payout = div_abs / abs(fcf) if fcf != 0 else None
    if fcf_payout and fcf_payout > 1.0:
        exclusions.append(f"FCF payout {fcf_payout:.0%} > 100%")

    # ── Hard Exclusion 3: Dividend Yield > 2× sector median ──────────────────
    forward_yield = None
    if div_per_share and price_per_share and price_per_share > 0:
        forward_yield = div_per_share / price_per_share

    # Store yield for sector comparison (done in score_universe)
    factors["_forward_yield"] = forward_yield

    # ── Hard Exclusion 4: No increase in last 3 years ─────────────────────────
    if len(fdf) >= 12:
        div_history = fdf["dividends_paid"].dropna().abs()
        if len(div_history) >= 2:
            recent_avg = div_history.head(4).mean()  # Recent year
            old_avg    = div_history.iloc[8:12].mean()   # 3 years ago
            if old_avg and old_avg > 0 and recent_avg <= old_avg:
                exclusions.append("No dividend increase in 3 years")
        else:
            exclusions.append("Insufficient dividend history")
    else:
        exclusions.append("Insufficient history for dividend trend")

    if exclusions:
        return factors, exclusions   # Return early — excluded

    # ── Safety factors ────────────────────────────────────────────────────────
    # Altman Z Safety (scaled 1.81–5.0 → 0–100)
    if z is not None:
        z_capped = min(max(z, ALTMAN_DISTRESS), 5.0)
        factors["altman_z_safety"] = (z_capped - ALTMAN_DISTRESS) / (5.0 - ALTMAN_DISTRESS) * 100
    else:
        factors["altman_z_safety"] = None

    # FCF Payout Ratio (inverted — lower payout = higher score)
    factors["fcf_payout_ratio_inv"] = (1 - fcf_payout) * 100 if (fcf_payout and fcf_payout < 1) else None

    # D/E Inverted
    de = safe_div(td, eq)
    factors["debt_to_equity_inv"] = (1 / de * 100) if (de and de > 0) else (100.0 if de == 0 else None)

    # Earnings Payout (inverted)
    earn_payout = safe_div(div_abs, abs(ni)) if ni else None
    factors["earnings_payout_inv"] = (1 - min(earn_payout, 1.0)) * 100 if earn_payout else None

    # ── Dividend Growth ───────────────────────────────────────────────────────
    div_history_abs = fdf["dividends_paid"].dropna().abs()
    def _div_cagr(n_quarters: int, years: float) -> Optional[float]:
        if len(div_history_abs) >= n_quarters:
            return compute_cagr([float(div_history_abs.iloc[-1]), float(div_history_abs.iloc[0])], years)
        return None

    factors["div_growth_5y"]  = _div_cagr(20, 5)
    factors["div_growth_10y"] = _div_cagr(40, 10)

    # Consecutive years of increases
    factors["consecutive_increases"] = _count_consecutive_increases(div_history_abs)

    # ── Yield ─────────────────────────────────────────────────────────────────
    factors["dividend_yield"] = forward_yield

    # Shareholder yield = (dividends + net buybacks) / market_cap
    buybacks = curr.get("capex")   # Simplified — net buybacks not in schema
    shareholder_yield = div_abs / market_cap if (div_abs and market_cap) else forward_yield
    factors["shareholder_yield"] = shareholder_yield

    # ── Value ─────────────────────────────────────────────────────────────────
    pe = safe_div(price_per_share, eps)
    factors["pe_ratio_inv"] = (1 / pe) if (pe and pe > 0) else None

    p_fcf = safe_div(market_cap, fcf) if fcf and fcf > 0 else None
    factors["p_fcf_inv"] = (1 / p_fcf) if (p_fcf and p_fcf > 0) else None

    ebitda = ebit  # Simplified (no depr in schema)
    ev_ebitda = safe_div(ev, ebitda)
    factors["ev_ebitda_inv"] = (1 / ev_ebitda) if (ev_ebitda and ev_ebitda > 0) else None

    # ── Quality ───────────────────────────────────────────────────────────────
    factors["roe"] = safe_div(ni, eq)
    factors["gross_profitability"] = safe_div(gp, ta)

    # Revenue growth (3-year CAGR)
    if len(fdf) >= 12:
        rev_now = rev or 0
        rev_3y  = fdf.iloc[11].get("revenue") or 0
        factors["revenue_growth"] = compute_cagr([rev_3y, rev_now], years=3) if rev_3y > 0 else None
    else:
        factors["revenue_growth"] = None

    # Chowder Rule flag (not in score — informational)
    div_growth_5y = factors.get("div_growth_5y") or 0
    chowder = (forward_yield or 0) + div_growth_5y
    chowder_threshold = CHOWDER_UTILITY if sector in UTILITY_SECTORS else CHOWDER_STANDARD
    factors["_chowder_rule_passes"] = chowder >= chowder_threshold
    factors["_chowder_value"] = chowder

    return factors, exclusions


def _count_consecutive_increases(div_series: pd.Series, cap: int = 25) -> float:
    """Count consecutive annual dividend increases from most recent data."""
    if len(div_series) < 2:
        return 0.0
    # Approximate annual by grouping every 4 quarters
    annuals = [div_series.iloc[i*4:(i+1)*4].mean() for i in range(len(div_series) // 4)]
    annuals = [a for a in annuals if pd.notna(a) and a > 0]
    if len(annuals) < 2:
        return 0.0
    count = 0
    for i in range(len(annuals) - 1):
        if annuals[i] >= annuals[i + 1]:
            count += 1
        else:
            break
    return float(min(count, cap))


def score_universe(as_of: Optional[str] = None, persist: bool = True) -> pd.DataFrame:
    """Score all active stocks on the dividend strategy."""
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
            raw["ticker"]    = tkr
            raw["stock_id"]  = sid
            raw["sector"]    = sector
            raw["_excluded"] = bool(excl)
            raw["_exclusion_reasons"] = "; ".join(excl) if excl else ""
            all_factors.append(raw)
        except Exception as e:
            log.debug("Dividend factor error for %s: %s", tkr, e)
            all_factors.append({"ticker": tkr, "stock_id": sid, "sector": sector, "_excluded": True, "_exclusion_reasons": str(e)})

    df = pd.DataFrame(all_factors)
    if df.empty:
        conn.close()
        return df

    # Hard-exclude stocks that fail filters
    df_excl = df[df["_excluded"]].copy()
    df_valid = df[~df["_excluded"]].copy()

    # Yield trap check: exclude yield > 2× sector median
    if not df_valid.empty and "_forward_yield" in df_valid.columns:
        sector_median_yield = df_valid.groupby("sector")["_forward_yield"].median()
        def _yield_trap(r):
            y = r.get("_forward_yield")
            s = r.get("sector")
            median = sector_median_yield.get(s, np.inf)
            return y is not None and median is not None and y > 2 * median
        is_yield_trap = df_valid.apply(_yield_trap, axis=1)
        df_trap = df_valid[is_yield_trap].copy()
        df_trap["_excluded"] = True
        df_trap["_exclusion_reasons"] += "; Yield > 2× sector median"
        df_valid = df_valid[~is_yield_trap].copy()
        df_excl = pd.concat([df_excl, df_trap], ignore_index=True)

    # Percentile rank factors cross-sectionally on eligible universe
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
    log.info("Dividend scoring complete. %d eligible, %d excluded.", len(df_valid), len(df_excl))
    return df_out
