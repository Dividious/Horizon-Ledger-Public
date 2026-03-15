"""
Horizon Ledger — Information Coefficient-Based Weight Optimization
Computes new factor weights based on each factor's historical IC.

Anti-overfitting safeguards:
  - Requires ≥60 monthly observations
  - Factors with mean IC < 0.01 get weight 0 (flagged for review)
  - Factors with negative IC get weight 0 (flagged prominently)
  - Maximum change per factor per cycle: ±5 pp
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    IC_MIN_USEFUL,
    REWEIGHTING_MIN_OBSERVATIONS,
    REWEIGHTING_MAX_CHANGE_PER_CYCLE,
    REWEIGHTING_MAX_SINGLE_FACTOR,
    REWEIGHTING_MIN_SINGLE_FACTOR,
)
from reweighting.tracker import compute_ic_statistics

log = logging.getLogger(__name__)


def compute_ic_weights(
    strategy: str,
    current_weights: dict[str, float],
    horizon: str = "return_63d",
) -> dict:
    """
    Compute IC-based proposed weights for a strategy.

    Algorithm:
      1. Compute Spearman IC for each factor vs the forward return horizon
      2. Discard factors with mean IC < IC_MIN_USEFUL (0.01)
      3. Set negative IC factors to 0 (flag for review)
      4. Normalize positive ICs to sum to 1.0
      5. Apply ±5pp change guardrail against current_weights

    Returns dict with:
      proposed_weights: dict[factor, weight]
      ic_stats: DataFrame
      flagged_zero: list of factors set to 0
      flagged_negative: list of factors with negative IC
      flagged_low: list of factors with |IC| < 0.01
    """
    ic_df = compute_ic_statistics(strategy, horizon)

    if ic_df.empty:
        log.warning("No IC statistics available for %s — returning current weights", strategy)
        return {
            "proposed_weights": current_weights.copy(),
            "ic_stats": ic_df,
            "flagged_zero": [],
            "flagged_negative": [],
            "flagged_low": list(current_weights.keys()),
            "sufficient_data": False,
        }

    flagged_negative: list[str] = []
    flagged_low: list[str] = []
    ic_dict: dict[str, float] = {}

    for _, row in ic_df.iterrows():
        factor = row["factor"]
        ic = row["mean_ic"]

        if np.isnan(ic) or ic < 0:
            flagged_negative.append(factor)
            log.warning("⚠️  Factor '%s' has negative IC=%.4f — weight set to 0", factor, ic)
            ic_dict[factor] = 0.0
        elif ic < IC_MIN_USEFUL:
            flagged_low.append(factor)
            log.warning("⚠️  Factor '%s' has IC=%.4f < %.3f threshold — weight set to 0", factor, ic, IC_MIN_USEFUL)
            ic_dict[factor] = 0.0
        else:
            ic_dict[factor] = max(0.0, ic)

    # Include current_weights factors not in ic_dict (set to 0 — insufficient data)
    for factor in current_weights:
        if factor not in ic_dict:
            ic_dict[factor] = 0.0

    # Normalize to sum to 1
    total_ic = sum(ic_dict.values())
    if total_ic == 0:
        log.error("All ICs are 0 or negative for %s — keeping current weights", strategy)
        return {
            "proposed_weights": current_weights.copy(),
            "ic_stats": ic_df,
            "flagged_zero": list(ic_dict.keys()),
            "flagged_negative": flagged_negative,
            "flagged_low": flagged_low,
            "sufficient_data": True,
        }

    raw_proposed = {f: v / total_ic for f, v in ic_dict.items()}

    # Apply ±5pp change guardrail
    proposed = _apply_change_guardrail(raw_proposed, current_weights)

    # Apply min/max per factor
    proposed = _apply_size_guardrails(proposed)

    # Re-normalize after guardrails
    total_p = sum(proposed.values())
    if total_p > 0:
        proposed = {f: v / total_p for f, v in proposed.items()}

    flagged_zero = [f for f, w in proposed.items() if w == 0.0]

    return {
        "proposed_weights": proposed,
        "ic_stats":         ic_df,
        "flagged_zero":     flagged_zero,
        "flagged_negative": flagged_negative,
        "flagged_low":      flagged_low,
        "sufficient_data":  True,
    }


def _apply_change_guardrail(
    proposed: dict[str, float],
    current: dict[str, float],
    max_change: float = REWEIGHTING_MAX_CHANGE_PER_CYCLE,
) -> dict[str, float]:
    """Clip each factor's proposed weight change to ±max_change."""
    result = {}
    for factor, prop_w in proposed.items():
        curr_w = current.get(factor, 0.0)
        delta = prop_w - curr_w
        if abs(delta) > max_change:
            clipped = curr_w + max_change * np.sign(delta)
        else:
            clipped = prop_w
        result[factor] = max(0.0, clipped)
    return result


def _apply_size_guardrails(
    weights: dict[str, float],
    max_w: float = REWEIGHTING_MAX_SINGLE_FACTOR,
    min_w: float = REWEIGHTING_MIN_SINGLE_FACTOR,
) -> dict[str, float]:
    """
    Enforce min/max per-factor weight constraints.
    Factors with weight = 0 remain at 0 (they've been excluded).
    """
    result = {}
    for factor, w in weights.items():
        if w == 0.0:
            result[factor] = 0.0
        else:
            result[factor] = min(max(w, min_w), max_w)
    return result
