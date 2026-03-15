"""
Horizon Ledger — Market Health Score + Digest Text Generator
Computes a composite 0-100 market health score from macro signals and
generates a plain-language research digest using conditional template logic.

NOT FINANCIAL ADVICE. All signals have historically produced false positives.
"""

import json
import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from config import (
    MARKET_HEALTH_WEIGHTS,
    CAPE_STRETCHED_THRESHOLD,
    CAPE_EXTREME_THRESHOLD,
)
from db.schema import get_connection
from db.queries import (
    get_current_regime,
    get_macro_series,
    upsert_digest,
    get_latest_digest,
)

log = logging.getLogger(__name__)


# ─── Sub-score Computation ────────────────────────────────────────────────────

def _score_regime(regime: Optional[str]) -> float:
    """bull=100, neutral=50, bear=0."""
    return {"bull": 100.0, "neutral": 50.0, "bear": 0.0}.get(regime or "neutral", 50.0)


def _score_yield_curve(slope: Optional[float]) -> float:
    """
    slope > 1.0  → 100
    0 to 1.0     → linear 50-100
    -0.5 to 0    → linear 20-50
    < -0.5       → 0
    """
    if slope is None:
        return 50.0
    if slope > 1.0:
        return 100.0
    elif slope >= 0.0:
        return 50.0 + (slope / 1.0) * 50.0
    elif slope >= -0.5:
        return 20.0 + ((slope + 0.5) / 0.5) * 30.0
    else:
        return 0.0


def _score_credit_spread(spread_bps: Optional[float]) -> float:
    """
    < 300 bps   → 100
    300-500     → linear 50-100
    500-800     → linear 10-50
    > 800       → 0
    """
    if spread_bps is None:
        return 50.0
    if spread_bps < 300:
        return 100.0
    elif spread_bps <= 500:
        return 50.0 + (500 - spread_bps) / 200.0 * 50.0
    elif spread_bps <= 800:
        return 10.0 + (800 - spread_bps) / 300.0 * 40.0
    else:
        return 0.0


def _score_vix(vix: Optional[float]) -> float:
    """
    < 15        → 100
    15-20       → linear 70-100
    20-30       → linear 30-70
    > 30        → 0
    """
    if vix is None:
        return 50.0
    if vix < 15.0:
        return 100.0
    elif vix <= 20.0:
        return 70.0 + (20.0 - vix) / 5.0 * 30.0
    elif vix <= 30.0:
        return 30.0 + (30.0 - vix) / 10.0 * 40.0
    else:
        return 0.0


def _score_sahm_rule(triggered: bool) -> float:
    """Not triggered = 100, triggered = 0."""
    return 0.0 if triggered else 100.0


def _score_pct_above_200sma(pct: Optional[float]) -> float:
    """
    > 70%       → 100
    50-70%      → linear 50-100
    30-50%      → linear 10-50
    < 30%       → 0
    """
    if pct is None:
        return 50.0
    if pct > 70.0:
        return 100.0
    elif pct >= 50.0:
        return 50.0 + (pct - 50.0) / 20.0 * 50.0
    elif pct >= 30.0:
        return 10.0 + (pct - 30.0) / 20.0 * 40.0
    else:
        return 0.0


def compute_market_health_score(
    regime: Optional[str] = None,
    yield_curve_slope: Optional[float] = None,
    credit_spread: Optional[float] = None,
    vix_level: Optional[float] = None,
    sahm_rule_value: Optional[float] = None,
    pct_above_200sma: Optional[float] = None,
) -> dict:
    """
    Compute the composite Market Health Score (0-100) from component signals.
    Returns dict with composite score, label, and per-component breakdown.
    """
    # Sahm Rule threshold: >= 0.5 = triggered
    sahm_triggered = sahm_rule_value is not None and sahm_rule_value >= 0.5

    sub_scores = {
        "regime":           _score_regime(regime),
        "yield_curve":      _score_yield_curve(yield_curve_slope),
        "credit_spread":    _score_credit_spread(credit_spread),
        "vix":              _score_vix(vix_level),
        "sahm_rule":        _score_sahm_rule(sahm_triggered),
        "pct_above_200sma": _score_pct_above_200sma(pct_above_200sma),
    }

    weights = MARKET_HEALTH_WEIGHTS
    composite = sum(sub_scores[k] * weights.get(k, 0.0) for k in sub_scores)

    if composite >= 65:
        label = "Favorable"
        emoji = "🟢"
    elif composite >= 40:
        label = "Caution"
        emoji = "🟡"
    else:
        label = "Risk-Off"
        emoji = "🔴"

    return {
        "composite_score": round(composite, 1),
        "label":           label,
        "emoji":           emoji,
        "sahm_triggered":  sahm_triggered,
        "sub_scores":      sub_scores,
        "weights":         weights,
    }


# ─── Auto-Generated Digest Text ──────────────────────────────────────────────

def generate_digest_text(
    regime: Optional[str],
    health_score: float,
    health_label: str,
    yield_curve_slope: Optional[float],
    yield_curve_inverted: bool,
    cape_ratio: Optional[float],
    cape_percentile: Optional[float],
    bubble_flags: Optional[dict],
    pct_above_200sma: Optional[float],
    sahm_rule_value: Optional[float],
    sahm_triggered: bool,
) -> str:
    """
    Generate a ~150-word market research digest using conditional template logic.
    No LLM calls — fully deterministic. Always ends with the mandatory disclaimer.
    NOT FINANCIAL ADVICE.
    """
    lines = []

    # ── Sentence 1: Regime and health label ──────────────────────────────────
    regime_str = (regime or "neutral").lower()
    lines.append(
        f"The HMM regime model currently indicates a {regime_str} market environment, "
        f"with an overall Market Health Score of {health_score:.0f}/100 ({health_label})."
    )

    # ── Sentence 2: Yield curve ───────────────────────────────────────────────
    if yield_curve_slope is not None:
        slope_bps = round(yield_curve_slope * 100)
        if yield_curve_inverted:
            inv_days = _count_inversion_days()
            lines.append(
                f"The yield curve is inverted at {slope_bps:+d} bps — "
                f"a pattern that has preceded recession in 7 of the last 8 historical instances; "
                f"the current inversion has persisted for approximately {inv_days} days."
            )
        else:
            lines.append(
                f"The yield curve is positive at {slope_bps:+d} bps (10Y minus 2Y), "
                "a constructive sign for near-term economic growth."
            )
    else:
        lines.append("Yield curve data is currently unavailable.")

    # ── Sentence 3: CAPE valuation ────────────────────────────────────────────
    if cape_ratio is not None and cape_percentile is not None:
        if cape_ratio > CAPE_EXTREME_THRESHOLD:
            val_word = "extremely elevated"
        elif cape_ratio > CAPE_STRETCHED_THRESHOLD:
            val_word = "elevated"
        elif cape_ratio > 22:
            val_word = "moderately elevated"
        elif cape_ratio > 15:
            val_word = "moderate"
        else:
            val_word = "depressed"

        lines.append(
            f"Market valuations are {val_word} with a Shiller CAPE of {cape_ratio:.1f} "
            f"({cape_percentile:.0f}th percentile historically)."
        )
    else:
        lines.append("CAPE valuation data is not yet available.")

    # ── Sentence 4: Active bubble flags (skip if none) ────────────────────────
    if bubble_flags:
        flag_keys = list(bubble_flags.keys())
        sector_flags = [k for k in flag_keys if k not in
                        ("market_stretched", "market_extreme", "credit_complacency",
                         "inversion_sustained")]
        mkt_flag = "market_extreme" in flag_keys or "market_stretched" in flag_keys
        credit_flag = "credit_complacency" in flag_keys

        parts = []
        if sector_flags:
            parts.append(f"sector valuation flags are active for: {', '.join(sector_flags)}")
        if mkt_flag:
            parts.append("overall market valuations are historically stretched")
        if credit_flag:
            parts.append("credit spreads are compressed (complacency signal)")

        if parts:
            lines.append("Risk flags: " + "; ".join(parts) + ".")

    # ── Sentence 5: Market breadth ────────────────────────────────────────────
    if pct_above_200sma is not None:
        if pct_above_200sma > 70:
            breadth_word = "broad"
        elif pct_above_200sma > 50:
            breadth_word = "moderate"
        elif pct_above_200sma > 30:
            breadth_word = "narrowing"
        else:
            breadth_word = "deteriorating"

        lines.append(
            f"{pct_above_200sma:.0f}% of universe stocks are trading above their "
            f"200-day moving average, indicating {breadth_word} market participation."
        )
    else:
        lines.append("Market breadth data (% above 200-day SMA) is not yet computed.")

    # ── Final sentence: Sahm Rule ──────────────────────────────────────────────
    if sahm_rule_value is not None:
        if sahm_triggered:
            lines.append(
                f"The Sahm Rule recession indicator is currently triggered "
                f"(value: {sahm_rule_value:.2f}) — historically associated with the "
                "onset of recession within one to two quarters."
            )
        else:
            lines.append(
                f"The Sahm Rule recession indicator is not triggered "
                f"(current value: {sahm_rule_value:.2f}; trigger threshold: 0.50)."
            )
    else:
        lines.append("Sahm Rule data is not yet available.")

    # ── Mandatory disclaimer ───────────────────────────────────────────────────
    lines.append(
        "This digest is a heuristic summary for research purposes only and is not "
        "financial advice. All signals have historically produced false positives."
    )

    return "  ".join(lines)


def _count_inversion_days() -> int:
    """Count consecutive days the yield curve has been inverted."""
    try:
        conn = get_connection()
        df = get_macro_series(conn, "DGS10",
                              start=(date.today() - timedelta(days=730)).isoformat())
        df2 = get_macro_series(conn, "DGS2",
                               start=(date.today() - timedelta(days=730)).isoformat())
        conn.close()
        if df.empty or df2.empty:
            return 0
        merged = df.rename(columns={"value": "dgs10"}).merge(
            df2.rename(columns={"value": "dgs2"}), on="date"
        )
        merged["slope"] = merged["dgs10"] - merged["dgs2"]
        merged = merged.sort_values("date", ascending=False)
        count = 0
        for _, r in merged.iterrows():
            if r["slope"] < 0:
                count += 1
            else:
                break
        return count
    except Exception:
        return 0


# ─── Main Update Function ─────────────────────────────────────────────────────

def update_market_digest(as_of: Optional[str] = None) -> dict:
    """
    Compute the full market digest for today and persist to market_digest_history.
    Called daily by run_daily.py (after technicals and macro are updated).
    Returns the digest data dict.
    """
    if as_of is None:
        as_of = date.today().isoformat()

    conn = get_connection()
    start90 = (date.fromisoformat(as_of) - timedelta(days=90)).isoformat()

    def _latest_macro(series: str) -> Optional[float]:
        df = get_macro_series(conn, series, start=start90, end=as_of)
        return float(df.iloc[-1]["value"]) if not df.empty else None

    # ── Gather raw signals ────────────────────────────────────────────────────
    regime_row  = get_current_regime(conn)
    regime      = regime_row["regime"] if regime_row else None

    dgs10       = _latest_macro("DGS10")
    dgs2        = _latest_macro("DGS2")
    slope       = (dgs10 - dgs2) if (dgs10 and dgs2) else None
    inverted    = slope is not None and slope < 0

    credit_spread = _latest_macro("BAMLH0A0HYM2")
    vix_level     = _latest_macro("VIXCLS")
    sahm_value    = _latest_macro("SAHMREALTIME")
    tips_breakeven = _latest_macro("T10YIE")

    # CAPE (monthly, use last recorded value)
    from pipeline.macro import get_cape_stats
    cape_stats    = get_cape_stats(conn)
    cape_ratio    = cape_stats.get("cape_ratio")
    cape_pct      = cape_stats.get("cape_percentile")

    conn.close()

    # pct_above_200sma computed from technicals cache
    from pipeline.macro import compute_pct_above_200sma
    pct_above = compute_pct_above_200sma()

    # ── Health score ──────────────────────────────────────────────────────────
    health = compute_market_health_score(
        regime=regime,
        yield_curve_slope=slope,
        credit_spread=credit_spread,
        vix_level=vix_level,
        sahm_rule_value=sahm_value,
        pct_above_200sma=pct_above,
    )

    # ── Bubble flags ──────────────────────────────────────────────────────────
    try:
        from pipeline.bubble_detector import compute_bubble_flags
        bubble_flags = compute_bubble_flags(
            cape_ratio=cape_ratio,
            cape_percentile=cape_pct,
            credit_spread=credit_spread,
            yield_curve_inverted=inverted,
        )
    except Exception as e:
        log.warning("Bubble flags failed: %s", e)
        bubble_flags = {}

    # ── Digest text ───────────────────────────────────────────────────────────
    digest_text = generate_digest_text(
        regime=regime,
        health_score=health["composite_score"],
        health_label=health["label"],
        yield_curve_slope=slope,
        yield_curve_inverted=inverted,
        cape_ratio=cape_ratio,
        cape_percentile=cape_pct,
        bubble_flags=bubble_flags,
        pct_above_200sma=pct_above,
        sahm_rule_value=sahm_value,
        sahm_triggered=health["sahm_triggered"],
    )

    # ── Persist ───────────────────────────────────────────────────────────────
    data = {
        "date":                 as_of,
        "market_health_score":  int(health["composite_score"]),
        "market_health_label":  health["label"],
        "regime":               regime,
        "cape_ratio":           cape_ratio,
        "cape_percentile":      cape_pct,
        "yield_curve_slope":    slope,
        "yield_curve_inverted": int(inverted),
        "credit_spread":        credit_spread,
        "vix_level":            vix_level,
        "sahm_rule_value":      sahm_value,
        "sahm_rule_triggered":  int(health["sahm_triggered"]),
        "tips_breakeven":       tips_breakeven,
        "pct_above_200sma":     pct_above,
        "bubble_flags":         json.dumps(bubble_flags),
        "digest_text":          digest_text,
    }

    conn = get_connection()
    with conn:
        upsert_digest(conn, data)
        conn.commit()
    conn.close()

    log.info(
        "Market digest updated: score=%d (%s), regime=%s",
        health["composite_score"], health["label"], regime
    )
    return {**data, "health_detail": health, "cape_stats": cape_stats}
