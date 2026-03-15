"""
Horizon Ledger — Ensemble Weight Optimizer
Combines IC-weighting and ElasticNet with guardrails and bootstrap CIs.

Blend: 50% IC-weights + 50% ElasticNet weights.
Bootstrap confidence intervals: 1,000 resamples.
Auto-approval only if ALL changes < 2% AND all CI do not cross zero.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    REWEIGHTING_MIN_OBSERVATIONS,
    REWEIGHTING_AUTO_APPROVE_THRESHOLD,
    REWEIGHTING_MAX_CHANGE_PER_CYCLE,
    REWEIGHTING_MAX_SINGLE_FACTOR,
    REWEIGHTING_MIN_SINGLE_FACTOR,
    WFE_MIN_ACCEPTABLE,
)

log = logging.getLogger(__name__)

N_BOOTSTRAP = 1_000
BLEND_IC_WEIGHT = 0.5
BLEND_EN_WEIGHT = 0.5


def compute_ensemble_weights(
    strategy: str,
    current_weights: dict[str, float],
    horizon: str = "return_63d",
) -> dict:
    """
    Run IC + ElasticNet optimization and ensemble their outputs.

    Returns a comprehensive result dict suitable for proposal.py.
    """
    from reweighting.ic_weighting import compute_ic_weights
    from reweighting.elastic_net import compute_elastic_net_weights

    log.info("Computing IC weights for '%s'...", strategy)
    ic_result = compute_ic_weights(strategy, current_weights, horizon)

    log.info("Computing ElasticNet weights for '%s'...", strategy)
    en_result = compute_elastic_net_weights(strategy, current_weights, horizon)

    # Check data sufficiency
    if not ic_result.get("sufficient_data") and not en_result.get("sufficient_data"):
        msg = en_result.get("message", "Insufficient data for reweighting.")
        return {
            "sufficient_data": False,
            "message": msg,
            "proposed_weights": current_weights.copy(),
        }

    ic_weights = ic_result["proposed_weights"]
    en_weights = en_result["proposed_weights"]

    # Ensemble blend
    all_factors = set(current_weights.keys()) | set(ic_weights.keys()) | set(en_weights.keys())
    ensemble_raw = {}
    for f in all_factors:
        ic_w = ic_weights.get(f, 0.0)
        en_w = en_weights.get(f, 0.0)
        ensemble_raw[f] = BLEND_IC_WEIGHT * ic_w + BLEND_EN_WEIGHT * en_w

    # Apply guardrails
    ensemble_proposed = _apply_all_guardrails(ensemble_raw, current_weights)

    # Re-normalize
    total = sum(ensemble_proposed.values())
    if total > 0:
        ensemble_proposed = {f: v / total for f, v in ensemble_proposed.items()}

    # Bootstrap confidence intervals
    log.info("Computing bootstrap CIs (%d resamples)...", N_BOOTSTRAP)
    ci_dict = _bootstrap_confidence_intervals(strategy, ensemble_proposed, current_weights, horizon)

    # Determine auto-approval eligibility
    max_change = max(
        abs(ensemble_proposed.get(f, 0) - current_weights.get(f, 0))
        for f in all_factors
    )
    all_ci_positive = all(
        ci["lower"] > 0 or ci["proposed_weight"] == 0
        for ci in ci_dict.values()
    )
    wfe = en_result.get("wfe")
    wfe_ok = wfe is None or wfe >= WFE_MIN_ACCEPTABLE

    auto_approvable = (
        max_change < REWEIGHTING_AUTO_APPROVE_THRESHOLD
        and all_ci_positive
        and wfe_ok
    )

    # Turnover impact estimate: sum of absolute weight changes
    turnover_impact = sum(
        abs(ensemble_proposed.get(f, 0) - current_weights.get(f, 0))
        for f in all_factors
    ) / 2   # Each $ rotated appears as both a buy and sell

    # Recommendation
    if not wfe_ok:
        recommendation = "INVESTIGATE"
        recommendation_reason = f"Walk-Forward Efficiency {wfe:.3f} < {WFE_MIN_ACCEPTABLE}"
    elif ic_result.get("flagged_negative"):
        recommendation = "INVESTIGATE"
        recommendation_reason = f"Factors with negative IC: {ic_result['flagged_negative']}"
    elif max_change < 0.02:
        recommendation = "APPROVE"
        recommendation_reason = "All changes < 2% — minor adjustment"
    elif max_change < 0.05:
        recommendation = "APPROVE"
        recommendation_reason = "Changes within guardrails — recommend approval"
    else:
        recommendation = "MODIFY"
        recommendation_reason = "Review proposed changes before approval"

    return {
        "sufficient_data":        True,
        "strategy":               strategy,
        "current_weights":        current_weights,
        "proposed_weights":       ensemble_proposed,
        "ic_weights":             ic_weights,
        "elastic_net_weights":    en_weights,
        "confidence_intervals":   ci_dict,
        "ic_stats":               ic_result.get("ic_stats", pd.DataFrame()),
        "flagged_negative":       ic_result.get("flagged_negative", []),
        "flagged_low":            ic_result.get("flagged_low", []),
        "dropped_factors":        en_result.get("dropped_factors", []),
        "low_t_factors":          en_result.get("low_t_factors", []),
        "t_stats":                en_result.get("t_stats", {}),
        "en_alpha":               en_result.get("alpha"),
        "en_l1_ratio":            en_result.get("l1_ratio"),
        "en_oos_r2":              en_result.get("oos_r2"),
        "wfe":                    wfe,
        "estimated_turnover_impact": turnover_impact,
        "max_change":             max_change,
        "auto_approvable":        auto_approvable,
        "recommendation":         recommendation,
        "recommendation_reason":  recommendation_reason,
        "n_observations":         en_result.get("n_observations", 0),
    }


def _apply_all_guardrails(
    weights: dict[str, float],
    current: dict[str, float],
) -> dict[str, float]:
    result = {}
    for factor, w in weights.items():
        curr_w = current.get(factor, 0.0)
        delta = w - curr_w
        if abs(delta) > REWEIGHTING_MAX_CHANGE_PER_CYCLE:
            w = curr_w + REWEIGHTING_MAX_CHANGE_PER_CYCLE * np.sign(delta)
        if w > 0:
            w = min(max(w, REWEIGHTING_MIN_SINGLE_FACTOR), REWEIGHTING_MAX_SINGLE_FACTOR)
        result[factor] = max(0.0, w)
    return result


def _bootstrap_confidence_intervals(
    strategy: str,
    proposed_weights: dict[str, float],
    current_weights: dict[str, float],
    horizon: str,
    n_iter: int = N_BOOTSTRAP,
) -> dict[str, dict]:
    """
    Bootstrap 95% CI for each proposed weight.
    Resamples predictions with replacement N_BOOTSTRAP times.

    Returns: {factor: {lower: float, upper: float, mean: float, proposed_weight: float}}
    """
    import json
    from db.schema import get_connection
    from db.queries import get_predictions_with_returns
    from scipy import stats

    conn = get_connection()
    preds = get_predictions_with_returns(conn, strategy, horizon)
    conn.close()

    if preds.empty or len(preds) < 20:
        return {
            f: {"lower": 0.0, "upper": 1.0, "mean": w, "proposed_weight": w}
            for f, w in proposed_weights.items()
        }

    def parse_comp(row):
        try:
            return json.loads(row.get("score_components", "{}") or "{}")
        except Exception:
            return {}

    preds = preds.copy()
    preds["components"] = preds.apply(parse_comp, axis=1)
    preds = preds.dropna(subset=[horizon])
    n = len(preds)

    boot_ics: dict[str, list[float]] = {f: [] for f in proposed_weights}

    rng = np.random.default_rng(seed=42)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        sample = preds.iloc[idx]

        for factor in proposed_weights:
            factor_vals = sample["components"].apply(lambda d: d.get(factor, 50.0))
            ret_vals    = sample[horizon]
            valid = pd.DataFrame({"f": factor_vals, "r": ret_vals}).dropna()
            if len(valid) < 5:
                continue
            ic, _ = stats.spearmanr(valid["f"], valid["r"])
            if not np.isnan(ic):
                boot_ics[factor].append(float(ic))

    ci_dict = {}
    for factor, w in proposed_weights.items():
        ics = boot_ics.get(factor, [])
        if len(ics) < 100:
            ci_dict[factor] = {"lower": -0.1, "upper": 0.1, "mean": 0.0, "proposed_weight": w}
        else:
            ics_arr = np.array(ics)
            ci_dict[factor] = {
                "lower":           float(np.percentile(ics_arr, 2.5)),
                "upper":           float(np.percentile(ics_arr, 97.5)),
                "mean":            float(ics_arr.mean()),
                "proposed_weight": w,
                "ci_crosses_zero": float(np.percentile(ics_arr, 2.5)) < 0 < float(np.percentile(ics_arr, 97.5)),
            }
    return ci_dict
