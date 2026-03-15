"""
Horizon Ledger — Shared Scoring Utilities
Percentile ranking, z-score normalization, and key financial signal calculators.

References:
  Piotroski (2000): Value investing — 9-signal F-Score
  Altman (1968): Financial ratios for bankruptcy prediction (Z-Score)
  Beneish (1999): Detecting earnings manipulation (M-Score)
  Novy-Marx (2013): Gross profitability factor
  Graham & Dodd: Graham Number
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from config import MISSING_FACTOR_FILL, ALTMAN_DISTRESS, BENEISH_THRESHOLD

log = logging.getLogger(__name__)


# ─── Cross-sectional normalization ───────────────────────────────────────────

def percentile_rank(series: pd.Series) -> pd.Series:
    """
    Rank each value in a cross-sectional series as a percentile (0–100).

    NaN handling: stocks with missing data receive MISSING_FACTOR_FILL (50th pct)
    — a neutral score that neither rewards nor penalizes incomplete data.
    This choice avoids systematic bias against smaller/newer companies with
    less historical data.  Document and review annually.
    """
    result = series.rank(pct=True) * 100
    result = result.fillna(MISSING_FACTOR_FILL)
    return result


def z_score_normalize(series: pd.Series, clip: float = 3.0) -> pd.Series:
    """
    Standard z-score normalization within a cross-section.
    Clips at ±clip to reduce outlier influence.
    """
    mu  = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    z = (series - mu) / std
    return z.clip(-clip, clip)


def weighted_composite_score(
    factor_scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    """
    Compute weighted sum of factor scores, normalized 0–100.
    Missing factors get MISSING_FACTOR_FILL.
    Weights are renormalized so they sum to 1 over available factors.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for factor, weight in weights.items():
        score = factor_scores.get(factor)
        if score is None or np.isnan(score):
            score = MISSING_FACTOR_FILL
        weighted_sum += weight * score
        total_weight += weight

    if total_weight == 0:
        return MISSING_FACTOR_FILL
    return weighted_sum / total_weight * (100 / 100)   # already 0–100 if inputs are percentiles


# ─── Piotroski F-Score ────────────────────────────────────────────────────────

def piotroski_f_score(curr: pd.Series, prev: Optional[pd.Series] = None) -> int:
    """
    Compute Piotroski (2000) F-Score (0–9).
    curr / prev: fundamentals rows (most recent and one year prior).

    9 binary signals:
    Profitability (4):
      F1: ROA > 0
      F2: Operating cash flow > 0
      F3: Change in ROA > 0
      F4: Accruals: CFO/Assets > ROA (cash earnings quality)
    Leverage / Liquidity (3):
      F5: Change in leverage (D/A) < 0
      F6: Change in current ratio > 0
      F7: No share dilution (shares outstanding did not increase)
    Operating efficiency (2):
      F8: Change in gross margin > 0
      F9: Change in asset turnover > 0
    """
    score = 0

    # Helper to safely get a float
    def g(row, key):
        if row is None:
            return None
        v = row.get(key) if isinstance(row, dict) else getattr(row, key, None)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)

    ta_curr = g(curr, "total_assets")
    ni_curr = g(curr, "net_income")
    ocf_curr = g(curr, "operating_cash_flow")
    rev_curr = g(curr, "revenue")
    gp_curr  = g(curr, "gross_profit")
    shares_curr = g(curr, "shares_outstanding")
    ca_curr  = g(curr, "current_assets")
    cl_curr  = g(curr, "current_liabilities")
    td_curr  = g(curr, "total_debt")

    # F1: ROA > 0
    if ta_curr and ta_curr > 0 and ni_curr is not None:
        roa_curr = ni_curr / ta_curr
        score += int(roa_curr > 0)

    # F2: CFO > 0
    if ocf_curr is not None:
        score += int(ocf_curr > 0)

    # F3 & F4 require prior year
    if prev is not None:
        ta_prev = g(prev, "total_assets")
        ni_prev = g(prev, "net_income")
        ocf_prev = g(prev, "operating_cash_flow")
        rev_prev = g(prev, "revenue")
        gp_prev  = g(prev, "gross_profit")
        shares_prev = g(prev, "shares_outstanding")
        ca_prev  = g(prev, "current_assets")
        cl_prev  = g(prev, "current_liabilities")
        td_prev  = g(prev, "total_debt")

        # F3: delta ROA > 0
        if ta_curr and ta_curr > 0 and ta_prev and ta_prev > 0 and ni_curr is not None and ni_prev is not None:
            roa_curr = ni_curr / ta_curr
            roa_prev = ni_prev / ta_prev
            score += int(roa_curr > roa_prev)

        # F4: CFO > ROA (accrual quality)
        if ta_curr and ta_curr > 0 and ni_curr is not None and ocf_curr is not None:
            roa_curr = ni_curr / ta_curr
            cfo_ratio = ocf_curr / ta_curr
            score += int(cfo_ratio > roa_curr)

        # F5: Change in leverage (LTD/Assets) < 0
        if ta_curr and ta_curr > 0 and ta_prev and ta_prev > 0 and td_curr is not None and td_prev is not None:
            lev_curr = td_curr / ta_curr
            lev_prev = td_prev / ta_prev
            score += int(lev_curr < lev_prev)

        # F6: Change in current ratio > 0
        if ca_curr and cl_curr and cl_curr > 0 and ca_prev and cl_prev and cl_prev > 0:
            cr_curr = ca_curr / cl_curr
            cr_prev = ca_prev / cl_prev
            score += int(cr_curr > cr_prev)

        # F7: No dilution
        if shares_curr is not None and shares_prev is not None:
            score += int(shares_curr <= shares_prev * 1.01)  # 1% tolerance

        # F8: Change in gross margin > 0
        if rev_curr and rev_curr > 0 and gp_curr is not None and rev_prev and rev_prev > 0 and gp_prev is not None:
            gm_curr = gp_curr / rev_curr
            gm_prev = gp_prev / rev_prev
            score += int(gm_curr > gm_prev)

        # F9: Change in asset turnover > 0
        if ta_curr and ta_curr > 0 and ta_prev and ta_prev > 0 and rev_curr is not None and rev_prev is not None:
            at_curr = rev_curr / ta_curr
            at_prev = rev_prev / ta_prev
            score += int(at_curr > at_prev)

    return score


def piotroski_to_percentile(f_score: int) -> float:
    """
    Convert F-Score (0–9) to percentile score using tiered approach.
    F >= 7 = 100 pts, 5-6 = 50 pts, < 5 = 0 pts.
    """
    if f_score >= 7:
        return 100.0
    elif f_score >= 5:
        return 50.0
    else:
        return 0.0


# ─── Altman Z-Score ──────────────────────────────────────────────────────────

def altman_z_score(
    working_capital: Optional[float],
    total_assets: Optional[float],
    retained_earnings: Optional[float],
    ebit: Optional[float],
    market_cap: Optional[float],
    total_liabilities: Optional[float],
    revenue: Optional[float],
) -> Optional[float]:
    """
    Altman (1968) Z-Score for public manufacturing firms.
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Cap / Total Liabilities
    X5 = Revenue / Total Assets

    Zones:
      Z > 2.99  → Safe (green)
      1.81–2.99 → Grey zone (yellow)
      Z < 1.81  → Distress (red)
    """
    if not total_assets or total_assets == 0:
        return None
    try:
        X1 = (working_capital or 0) / total_assets
        X2 = (retained_earnings or 0) / total_assets
        X3 = (ebit or 0) / total_assets
        X4 = (market_cap or 0) / (total_liabilities or 1)
        X5 = (revenue or 0) / total_assets
        z = 1.2 * X1 + 1.4 * X2 + 3.3 * X3 + 0.6 * X4 + 1.0 * X5
        return min(max(z, -5.0), 15.0)   # Cap for display purposes
    except Exception:
        return None


def altman_z_from_row(row: pd.Series, market_cap: Optional[float] = None) -> Optional[float]:
    """Convenience wrapper that extracts fields from a fundamentals row."""
    wc = (row.get("current_assets") or 0) - (row.get("current_liabilities") or 0)
    ta = row.get("total_assets")
    # Retained earnings not directly in our schema — approximate as total_equity - paid_in_capital
    # Use net_income TTM as a proxy if retained earnings unavailable
    re = row.get("total_equity")   # Simplified approximation
    ebit = row.get("ebit")
    tl = (row.get("total_assets") or 0) - (row.get("total_equity") or 0)
    rev = row.get("revenue")
    return altman_z_score(wc, ta, re, ebit, market_cap, tl, rev)


# ─── Beneish M-Score ─────────────────────────────────────────────────────────

def beneish_m_score(curr: pd.Series, prev: pd.Series) -> Optional[float]:
    """
    Beneish (1999) M-Score for earnings manipulation detection.
    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
         + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

    M > -1.78 → likely manipulator → EXCLUDE from all strategies.

    Requires 2 consecutive periods of data.
    """
    try:
        def g(row, key):
            v = row.get(key) if isinstance(row, dict) else getattr(row, key, None)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            return float(v)

        rev_t  = g(curr, "revenue");          rev_t1  = g(prev, "revenue")
        ar_t   = g(curr, "current_assets");   ar_t1   = g(prev, "current_assets")  # proxy for AR
        gp_t   = g(curr, "gross_profit");     gp_t1   = g(prev, "gross_profit")
        ta_t   = g(curr, "total_assets");     ta_t1   = g(prev, "total_assets")
        ni_t   = g(curr, "net_income");       ni_t1   = g(prev, "net_income")
        ocf_t  = g(curr, "operating_cash_flow")
        td_t   = g(curr, "total_debt");       td_t1   = g(prev, "total_debt")
        cl_t   = g(curr, "current_liabilities"); cl_t1 = g(prev, "current_liabilities")
        sga_t  = None  # SG&A not always separately available — skip SGAI

        # DSRI: Days Sales Receivable Index (higher → inflated receivables)
        dsri = None
        if ar_t and rev_t and ar_t1 and rev_t1 and rev_t > 0 and rev_t1 > 0:
            dsri = (ar_t / rev_t) / (ar_t1 / rev_t1)

        # GMI: Gross Margin Index
        gmi = None
        if gp_t and rev_t and gp_t1 and rev_t1 and rev_t > 0 and rev_t1 > 0:
            gm_t  = gp_t  / rev_t
            gm_t1 = gp_t1 / rev_t1
            gmi = gm_t1 / gm_t if gm_t != 0 else None

        # AQI: Asset Quality Index
        aqi = None
        if ta_t and ta_t1 and ta_t > 0 and ta_t1 > 0:
            # Non-current, non-physical assets / total assets (simplified)
            ca_t  = g(curr, "current_assets") or 0
            ca_t1 = g(prev, "current_assets") or 0
            ppe_proxy = 0   # PPE not in our schema — simplified
            nqa_t  = (ta_t  - ca_t  - ppe_proxy) / ta_t
            nqa_t1 = (ta_t1 - ca_t1 - ppe_proxy) / ta_t1
            aqi = nqa_t / nqa_t1 if nqa_t1 != 0 else None

        # SGI: Sales Growth Index
        sgi = None
        if rev_t and rev_t1 and rev_t1 > 0:
            sgi = rev_t / rev_t1

        # DEPI: Depreciation Index (not in schema — skip / set to 1)
        depi = 1.0

        # SGAI: SG&A Index (not available — skip / set to 1)
        sgai = 1.0

        # LVGI: Leverage Index
        lvgi = None
        if ta_t and ta_t1 and ta_t > 0 and ta_t1 > 0:
            lev_t  = ((td_t  or 0) + (cl_t  or 0)) / ta_t
            lev_t1 = ((td_t1 or 0) + (cl_t1 or 0)) / ta_t1
            lvgi = lev_t / lev_t1 if lev_t1 != 0 else None

        # TATA: Total Accruals to Total Assets
        tata = None
        if ta_t and ta_t > 0 and ni_t is not None and ocf_t is not None:
            tata = (ni_t - ocf_t) / ta_t

        # Apply formula with fallback to 1 for unavailable components
        m = (
            -4.84
            + 0.920 * (dsri  or 1.0)
            + 0.528 * (gmi   or 1.0)
            + 0.404 * (aqi   or 1.0)
            + 0.892 * (sgi   or 1.0)
            + 0.115 * depi
            - 0.172 * sgai
            + 4.679 * (tata  or 0.0)
            - 0.327 * (lvgi  or 1.0)
        )
        return m
    except Exception as e:
        log.debug("Beneish M-Score error: %s", e)
        return None


def beneish_screen_score(m: Optional[float]) -> float:
    """
    Convert M-Score to a 0/0.5/1.0 screening signal.
    > -1.78  → 0.0 (DISQUALIFY — likely manipulation)
    -2.22 to -1.78 → 0.5 (caution)
    < -2.22  → 1.0 (clean)
    Then convert to percentile (0/50/100).
    """
    if m is None:
        return MISSING_FACTOR_FILL
    if m > BENEISH_THRESHOLD:    # > -1.78
        return 0.0
    elif m > -2.22:
        return 50.0
    else:
        return 100.0


# ─── Enterprise Value ─────────────────────────────────────────────────────────

def compute_enterprise_value(
    market_cap: Optional[float],
    total_debt: Optional[float],
    cash: Optional[float],
) -> Optional[float]:
    """
    EV = Market Cap + Total Debt - Cash
    Standard enterprise value calculation.
    """
    if market_cap is None:
        return None
    return market_cap + (total_debt or 0) - (cash or 0)


# ─── Graham Number ────────────────────────────────────────────────────────────

def graham_number(eps: Optional[float], book_value: Optional[float]) -> Optional[float]:
    """
    Graham Number = sqrt(22.5 * EPS * Book Value Per Share)
    Represents the maximum intrinsic value a defensive investor should pay.
    Only valid when both EPS and BVPS are positive.
    """
    if eps and book_value and eps > 0 and book_value > 0:
        return math.sqrt(22.5 * eps * book_value)
    return None


# ─── CAGR ─────────────────────────────────────────────────────────────────────

def compute_cagr(values: list[float], years: float) -> Optional[float]:
    """
    Compound Annual Growth Rate over `years` from a list of ordered values.
    Returns None if inputs are invalid.
    """
    if len(values) < 2 or years <= 0:
        return None
    start = values[0]
    end   = values[-1]
    if start is None or end is None or start == 0:
        return None
    try:
        return (end / start) ** (1 / years) - 1
    except Exception:
        return None


# ─── Safe ratio helper ────────────────────────────────────────────────────────

def safe_div(numerator, denominator, default=None):
    """Safe division returning default when denominator is zero or None."""
    if denominator is None or denominator == 0:
        return default
    if numerator is None:
        return default
    return numerator / denominator
