"""
Horizon Ledger — Hidden Markov Model Regime Detector
3-state Gaussian HMM on macro features.
States: bull | neutral | bear

Features:
  sp500_monthly_return, yield_curve_slope, credit_spread, vix_level

Retrained quarterly on 10 years of monthly history.
Reference: Ang & Bekaert (2002), Hamilton (1989) HMM for regime detection.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    BASE_DIR,
    HMM_N_STATES,
    HMM_HISTORY_YEARS,
    HMM_REGIME_LABELS,
)
from db.schema import get_connection
from db.queries import upsert_regime, get_current_regime

log = logging.getLogger(__name__)

MODEL_PATH = BASE_DIR / "data" / "hmm_model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = ["sp500_monthly_return", "yield_curve_slope", "credit_spread", "vix"]
REGIME_ORDER = ["bear", "neutral", "bull"]    # Assigned by mean return of each state


def _label_states_by_return(model, feature_matrix: np.ndarray) -> dict[int, str]:
    """
    Map hidden states to economic labels by sorting on mean SP500 return.
    State with lowest mean return = bear, highest = bull.
    """
    states = model.predict(feature_matrix)
    # Use first feature (sp500 return) to order states
    means = {}
    for s in range(HMM_N_STATES):
        mask = states == s
        if mask.any():
            means[s] = feature_matrix[mask, 0].mean()
        else:
            means[s] = 0.0
    sorted_states = sorted(means.keys(), key=lambda s: means[s])
    return {sorted_states[0]: "bear", sorted_states[1]: "neutral", sorted_states[2]: "bull"}


def train_hmm(retrain: bool = False) -> object:
    """
    Train (or load cached) a 3-state Gaussian HMM on macro features.
    Returns the fitted model.
    """
    if MODEL_PATH.exists() and not retrain:
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            log.warning("Failed to load cached HMM model — retraining")

    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        log.error("hmmlearn not installed. Install it with: pip install hmmlearn")
        return None

    from pipeline.macro import get_macro_history_for_hmm
    df = get_macro_history_for_hmm(HMM_HISTORY_YEARS)

    if df.empty or len(df) < 24:
        log.warning("Insufficient macro history to train HMM (%d months)", len(df))
        return None

    # Fill missing features with rolling mean
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols].values.astype(float)

    # Remove rows with NaN
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]

    if len(X) < 24:
        log.warning("Too few valid rows after NaN removal: %d", len(X))
        return None

    log.info("Training HMM on %d monthly observations (%d features)...", len(X), X.shape[1])
    model = GaussianHMM(
        n_components=HMM_N_STATES,
        covariance_type="full",
        n_iter=500,
        random_state=42,
        tol=1e-4,
    )
    model.fit(X)

    # Attach label mapping to model for later use
    model._label_map = _label_states_by_return(model, X)
    model._feature_cols = available_cols

    # Cache model
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        log.info("HMM model saved to %s", MODEL_PATH)
    except Exception as e:
        log.warning("Could not save HMM model: %s", e)

    return model


def predict_current_regime(
    model=None,
    features: Optional[dict] = None,
) -> dict:
    """
    Predict the current market regime using the latest macro features.
    Returns: {regime, prob_bear, prob_neutral, prob_bull, confidence}
    """
    if model is None:
        model = train_hmm()
    if model is None:
        return {"regime": "unknown", "prob_bear": 0.33, "prob_neutral": 0.34, "prob_bull": 0.33}

    if features is None:
        from pipeline.macro import get_macro_features
        features = get_macro_features()

    label_map = getattr(model, "_label_map", {0: "bear", 1: "neutral", 2: "bull"})
    feat_cols  = getattr(model, "_feature_cols", FEATURE_COLS)

    X = np.array([[features.get(c, 0.0) or 0.0 for c in feat_cols]])

    try:
        posteriors = model.predict_proba(X)[0]  # Shape: (n_states,)
        state = int(np.argmax(posteriors))
        regime_label = label_map.get(state, "neutral")

        # Map state probs to named regimes
        bear_prob    = sum(posteriors[s] for s, l in label_map.items() if l == "bear")
        neutral_prob = sum(posteriors[s] for s, l in label_map.items() if l == "neutral")
        bull_prob    = sum(posteriors[s] for s, l in label_map.items() if l == "bull")

        return {
            "regime":       regime_label,
            "prob_bear":    float(bear_prob),
            "prob_neutral": float(neutral_prob),
            "prob_bull":    float(bull_prob),
            "confidence":   float(max(posteriors)),
        }
    except Exception as e:
        log.error("HMM prediction failed: %s", e)
        return {"regime": "unknown", "prob_bear": 0.33, "prob_neutral": 0.34, "prob_bull": 0.33}


def update_regime_history(retrain: bool = False) -> dict:
    """
    Predict and store the current regime in the DB.
    Call quarterly (retrain=True) or weekly.
    """
    from datetime import date

    model = train_hmm(retrain=retrain)
    regime_info = predict_current_regime(model)
    today = date.today().isoformat()

    conn = get_connection()
    with conn:
        upsert_regime(
            conn, today,
            regime_info["regime"],
            regime_info["prob_bear"],
            regime_info["prob_neutral"],
            regime_info["prob_bull"],
        )
        conn.commit()
    conn.close()

    log.info(
        "Current regime: %s (bear=%.1f%%, neutral=%.1f%%, bull=%.1f%%)",
        regime_info["regime"],
        regime_info["prob_bear"] * 100,
        regime_info["prob_neutral"] * 100,
        regime_info["prob_bull"] * 100,
    )
    return regime_info


def get_regime_factor_context(regime: str) -> str:
    """
    Return human-readable guidance on which factors tend to work in this regime.
    Used in reweighting proposals for context.
    """
    guidance = {
        "bull": (
            "In bull regimes, momentum factors (momentum_6m, sector_momentum) "
            "and growth factors (revenue_cagr_5y, earnings_cagr_5y) tend to outperform. "
            "Consider upweighting momentum vs. quality."
        ),
        "bear": (
            "In bear regimes, quality and safety factors (altman_z, gross_profitability, "
            "current_ratio, debt_to_equity_inv) tend to provide better downside protection. "
            "Consider upweighting quality vs. momentum."
        ),
        "neutral": (
            "In neutral regimes, default weight configuration applies. "
            "No strong regime tilt recommended."
        ),
        "unknown": "Regime could not be determined — applying default weights.",
    }
    return guidance.get(regime, guidance["neutral"])
