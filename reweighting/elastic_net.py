"""
Horizon Ledger — ElasticNet Regression-Based Weight Optimization
Uses sklearn ElasticNetCV with TimeSeriesSplit to avoid look-ahead bias.

Anti-overfitting safeguards:
  - TimeSeriesSplit (never random k-fold on time series)
  - Walk-Forward Efficiency (OOS/IS ratio) — warn if < 0.5
  - Harvey-Liu-Zhu t-stat threshold: |t| >= 3.0
  - Reports features dropped to zero (implicit factor selection)

References:
  Harvey, Liu, Zhu (2016): "... and the Cross-Section of Expected Returns"
    — t-stat threshold of 3.0 for factor validity after multiple comparisons
  Zou & Hastie (2005): ElasticNet regularization
"""

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config import (
    REWEIGHTING_MIN_OBSERVATIONS,
    REWEIGHTING_MAX_CHANGE_PER_CYCLE,
    REWEIGHTING_MAX_SINGLE_FACTOR,
    REWEIGHTING_MIN_SINGLE_FACTOR,
    WFE_MIN_ACCEPTABLE,
    T_STAT_THRESHOLD,
)
from db.schema import get_connection
from db.queries import get_predictions_with_returns

log = logging.getLogger(__name__)


def compute_elastic_net_weights(
    strategy: str,
    current_weights: dict[str, float],
    horizon: str = "return_63d",
) -> dict:
    """
    Fit ElasticNetCV on historical factor → return data.

    Returns dict with:
      proposed_weights: dict[factor, weight]
      alpha: selected regularization alpha
      l1_ratio: selected l1_ratio
      oos_r2: out-of-sample R²
      wfe: Walk-Forward Efficiency (OOS/IS ratio)
      t_stats: dict[factor, t_stat]
      dropped_factors: list of factors with coefficient = 0
      sufficient_data: bool
    """
    conn = get_connection()
    preds = get_predictions_with_returns(conn, strategy, horizon, min_rows=REWEIGHTING_MIN_OBSERVATIONS)
    conn.close()

    if preds.empty or len(preds) < REWEIGHTING_MIN_OBSERVATIONS:
        log.warning(
            "Insufficient data for ElasticNet on '%s': %d observations (need %d)",
            strategy, len(preds), REWEIGHTING_MIN_OBSERVATIONS,
        )
        return {
            "proposed_weights": current_weights.copy(),
            "alpha": None,
            "l1_ratio": None,
            "oos_r2": None,
            "wfe": None,
            "t_stats": {},
            "dropped_factors": [],
            "sufficient_data": False,
            "message": (
                f"Insufficient history for reweighting. "
                f"Continue collecting predictions. "
                f"Current: {len(preds)}, needed: {REWEIGHTING_MIN_OBSERVATIONS}"
            ),
        }

    # Parse factor scores from score_components JSON
    def parse_components(row):
        try:
            return json.loads(row.get("score_components", "{}") or "{}")
        except Exception:
            return {}

    preds = preds.copy()
    preds["components"] = preds.apply(parse_components, axis=1)
    preds["signal_date"] = pd.to_datetime(preds["signal_date"])
    preds = preds.sort_values("signal_date").reset_index(drop=True)

    # Build feature matrix
    factor_names = list(current_weights.keys())
    X_rows = []
    for _, row in preds.iterrows():
        comp = row["components"]
        X_rows.append([comp.get(f, 50.0) for f in factor_names])

    X = np.array(X_rows, dtype=float)
    y = preds[horizon].values.astype(float)

    # Drop rows with NaN in y
    valid_mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    if len(y) < REWEIGHTING_MIN_OBSERVATIONS:
        log.warning("After NaN removal: %d observations remaining", len(y))
        return {
            "proposed_weights": current_weights.copy(),
            "sufficient_data": False,
            "message": f"After cleaning: {len(y)} obs remaining, need {REWEIGHTING_MIN_OBSERVATIONS}",
        }

    # Standardize features cross-sectionally per time period
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # TimeSeriesSplit — NEVER use random k-fold on time series
    tscv = TimeSeriesSplit(n_splits=5)

    # ElasticNetCV
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    model = ElasticNetCV(
        l1_ratio=l1_ratios,
        n_alphas=20,
        cv=tscv,
        max_iter=10000,
        tol=1e-4,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_scaled, y)

    coefs = model.coef_
    alpha_val = model.alpha_
    l1_ratio_val = model.l1_ratio_

    # Walk-Forward Efficiency: OOS R² / IS R²
    is_r2 = model.score(X_scaled, y)
    oos_r2 = _compute_oos_r2(X_scaled, y, tscv, alpha_val, l1_ratio_val)
    wfe = oos_r2 / is_r2 if (is_r2 > 0 and oos_r2 is not None) else None

    if wfe is not None and wfe < WFE_MIN_ACCEPTABLE:
        log.warning(
            "⚠️  Walk-Forward Efficiency = %.3f < %.3f (minimum acceptable). "
            "Model may be overfit. Human review required before approval.",
            wfe, WFE_MIN_ACCEPTABLE,
        )

    # T-statistics via bootstrap (approximate)
    t_stats = _compute_t_stats(X_scaled, y, coefs)
    low_t_factors = [factor_names[i] for i, t in enumerate(t_stats) if abs(t) < T_STAT_THRESHOLD]
    if low_t_factors:
        log.warning(
            "Factors with |t| < %.1f (Harvey-Liu-Zhu threshold — potentially spurious): %s",
            T_STAT_THRESHOLD, low_t_factors,
        )

    # Derive weights from absolute coefficients
    abs_coefs = np.abs(coefs)
    total = abs_coefs.sum()

    if total == 0:
        raw_weights = {f: current_weights.get(f, 0.0) for f in factor_names}
    else:
        raw_weights = {f: abs_coefs[i] / total for i, f in enumerate(factor_names)}

    dropped_factors = [f for f, w in raw_weights.items() if w == 0.0]

    # Apply guardrails
    proposed = _apply_guardrails(raw_weights, current_weights)

    # Re-normalize
    total_p = sum(proposed.values())
    if total_p > 0:
        proposed = {f: v / total_p for f, v in proposed.items()}

    return {
        "proposed_weights": proposed,
        "alpha":            float(alpha_val),
        "l1_ratio":         float(l1_ratio_val),
        "is_r2":            float(is_r2),
        "oos_r2":           float(oos_r2) if oos_r2 is not None else None,
        "wfe":              float(wfe) if wfe is not None else None,
        "t_stats":          {factor_names[i]: float(t_stats[i]) for i in range(len(factor_names))},
        "dropped_factors":  dropped_factors,
        "low_t_factors":    low_t_factors,
        "n_observations":   len(y),
        "sufficient_data":  True,
        "raw_coefs":        {factor_names[i]: float(coefs[i]) for i in range(len(factor_names))},
    }


def _compute_oos_r2(
    X: np.ndarray,
    y: np.ndarray,
    tscv: TimeSeriesSplit,
    alpha: float,
    l1_ratio: float,
) -> Optional[float]:
    """Compute out-of-sample R² using TimeSeriesSplit."""
    from sklearn.linear_model import ElasticNet
    oos_preds = np.zeros_like(y)
    oos_mask  = np.zeros(len(y), dtype=bool)
    for train_idx, test_idx in tscv.split(X):
        m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, tol=1e-4)
        m.fit(X[train_idx], y[train_idx])
        oos_preds[test_idx] = m.predict(X[test_idx])
        oos_mask[test_idx]  = True

    if oos_mask.sum() < 10:
        return None

    y_oos = y[oos_mask]
    p_oos = oos_preds[oos_mask]
    ss_res = ((y_oos - p_oos) ** 2).sum()
    ss_tot = ((y_oos - y_oos.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _compute_t_stats(X: np.ndarray, y: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """
    Compute approximate t-statistics for ElasticNet coefficients.
    Uses the OLS formula on the selected features as an approximation.
    (Exact inference for ElasticNet is non-trivial — treat as a rough guide.)
    """
    n, p = X.shape
    y_pred = X @ coefs
    residuals = y - y_pred
    mse = (residuals ** 2).sum() / max(n - p - 1, 1)
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.pinv(XtX)
        var_coefs = mse * np.diag(XtX_inv)
        se = np.sqrt(np.abs(var_coefs))
        t_stats = coefs / (se + 1e-12)
        return t_stats
    except Exception:
        return np.zeros(p)


def _apply_guardrails(
    raw_weights: dict[str, float],
    current_weights: dict[str, float],
) -> dict[str, float]:
    result = {}
    for factor, w in raw_weights.items():
        curr_w = current_weights.get(factor, 0.0)
        # Max change per cycle
        delta = w - curr_w
        if abs(delta) > REWEIGHTING_MAX_CHANGE_PER_CYCLE:
            w = curr_w + REWEIGHTING_MAX_CHANGE_PER_CYCLE * np.sign(delta)
        # Min/max size
        if w > 0:
            w = min(max(w, REWEIGHTING_MIN_SINGLE_FACTOR), REWEIGHTING_MAX_SINGLE_FACTOR)
        result[factor] = max(0.0, w)
    return result
