"""
Horizon Ledger — Prediction Tracking & Forward Return Fill-In
Core feedback mechanism that enables the system to measure its own accuracy.

Every week, scoring runs and logs predictions.
This module fills in realized forward returns as time passes.
The filled predictions feed into the IC-weighting and ElasticNet reweighting.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from db.schema import get_connection
from db.queries import (
    get_unfilled_predictions,
    fill_prediction_return,
    get_predictions_with_returns,
)
from pipeline.prices import compute_forward_return

log = logging.getLogger(__name__)

# Forward horizons: (field_name, trading_days_approx, calendar_days_conservative)
HORIZONS = [
    ("return_5d",   5,   7),
    ("return_21d",  21,  30),
    ("return_63d",  63,  90),
    ("return_126d", 126, 180),
    ("return_252d", 252, 365),
]


def fill_forward_returns(as_of: Optional[str] = None) -> dict[str, int]:
    """
    For each prediction that is old enough, fill in the realized forward return.
    Returns dict of {horizon_field: count_filled}.
    """
    if as_of is None:
        as_of = date.today().isoformat()

    conn = get_connection()
    filled_counts: dict[str, int] = {}

    for field, _, cal_days in HORIZONS:
        unfilled = get_unfilled_predictions(conn, field, cal_days, as_of)
        if unfilled.empty:
            filled_counts[field] = 0
            continue

        count = 0
        with conn:
            for _, row in unfilled.iterrows():
                ticker      = row["ticker"]
                signal_date = row["signal_date"]
                pred_id     = row["id"]

                ret = compute_forward_return(ticker, signal_date, cal_days)
                if ret is not None:
                    fill_prediction_return(conn, pred_id, field, ret)
                    count += 1
            conn.commit()

        filled_counts[field] = count
        log.info("Filled %s returns: %d predictions updated", field, count)

    conn.close()
    return filled_counts


def compute_ic_statistics(
    strategy: str,
    horizon: str = "return_63d",
    window_months: int = 12,
) -> pd.DataFrame:
    """
    Compute Information Coefficient (IC) statistics per factor.

    IC = Spearman rank correlation between factor score at signal date
         and realized forward return.

    Returns DataFrame with columns:
      factor, mean_ic, std_ic, ic_ir, hit_rate, trailing_6m_ic, trailing_12m_ic
    """
    conn = get_connection()
    preds = get_predictions_with_returns(conn, strategy, horizon)
    conn.close()

    if preds.empty:
        log.warning("No predictions with returns for strategy '%s'", strategy)
        return pd.DataFrame()

    import json
    import numpy as np
    from scipy import stats

    # Parse score_components JSON
    def parse_components(row):
        try:
            return json.loads(row.get("score_components", "{}") or "{}")
        except Exception:
            return {}

    preds["components"] = preds.apply(parse_components, axis=1)
    preds["signal_date"] = pd.to_datetime(preds["signal_date"])

    # Get all factor names from components
    all_factors = set()
    for comp in preds["components"]:
        all_factors.update(comp.keys())
    all_factors.discard("")

    rows = []
    for factor in sorted(all_factors):
        factor_vals = preds["components"].apply(lambda d: d.get(factor))
        returns     = preds[horizon]

        # Drop NaN pairs
        valid = pd.DataFrame({"factor": factor_vals, "ret": returns}).dropna()
        if len(valid) < 20:
            continue

        # Full-period Spearman IC
        ic_full, _ = stats.spearmanr(valid["factor"], valid["ret"])

        # Rolling 12-month IC
        preds_copy = preds.copy()
        preds_copy["factor_val"] = factor_vals
        preds_copy = preds_copy.dropna(subset=["factor_val", horizon])
        preds_copy = preds_copy.sort_values("signal_date")

        cutoff_12m = preds_copy["signal_date"].max() - pd.DateOffset(months=12)
        cutoff_6m  = preds_copy["signal_date"].max() - pd.DateOffset(months=6)

        trail_12m_df = preds_copy[preds_copy["signal_date"] >= cutoff_12m]
        trail_6m_df  = preds_copy[preds_copy["signal_date"] >= cutoff_6m]

        ic_12m = float(stats.spearmanr(trail_12m_df["factor_val"], trail_12m_df[horizon])[0]) if len(trail_12m_df) >= 10 else np.nan
        ic_6m  = float(stats.spearmanr(trail_6m_df["factor_val"],  trail_6m_df[horizon])[0])  if len(trail_6m_df)  >= 10 else np.nan

        # Monthly rolling IC series for IC_IR
        preds_copy["month"] = preds_copy["signal_date"].dt.to_period("M")
        monthly_ics = []
        for _, m_df in preds_copy.groupby("month"):
            if len(m_df) < 5:
                continue
            ic_m, _ = stats.spearmanr(m_df["factor_val"], m_df[horizon])
            if not np.isnan(ic_m):
                monthly_ics.append(ic_m)

        ic_ir = float(np.mean(monthly_ics) / np.std(monthly_ics)) if len(monthly_ics) >= 6 and np.std(monthly_ics) > 0 else np.nan

        # Hit rate: fraction of top-quartile predictions that beat median return
        q75 = valid["factor"].quantile(0.75)
        top_q = valid[valid["factor"] >= q75]
        median_ret = valid["ret"].median()
        hit_rate = float((top_q["ret"] > median_ret).mean()) if len(top_q) > 0 else np.nan

        rows.append({
            "factor":        factor,
            "mean_ic":       float(ic_full),
            "ic_12m":        ic_12m,
            "ic_6m":         ic_6m,
            "ic_ir":         ic_ir,
            "hit_rate":      hit_rate,
            "n_obs":         len(valid),
            "is_useful":     abs(ic_full) >= 0.01,
            "trending_up":   (ic_6m > ic_12m) if (not np.isnan(ic_6m) and not np.isnan(ic_12m)) else None,
        })

    return pd.DataFrame(rows)


def store_ic_statistics(strategy: str, as_of: Optional[str] = None) -> pd.DataFrame:
    """
    Compute IC statistics for all factors in a strategy and persist to JSON.
    Called weekly from run_weekly.py after forward returns are filled.

    Returns DataFrame with IC stats per factor.
    """
    if as_of is None:
        as_of = date.today().isoformat()

    ic_df = compute_ic_statistics(strategy)

    if ic_df.empty:
        log.info("IC statistics for '%s': no data yet (need 20+ observations)", strategy)
        return ic_df

    # Log summary
    useful = ic_df[ic_df["is_useful"] == True]
    log.info(
        "IC statistics for '%s': %d factors, %d useful (IC≥0.01), mean_IC=%.4f",
        strategy, len(ic_df), len(useful), ic_df["mean_ic"].mean()
    )
    for _, r in ic_df.sort_values("mean_ic", ascending=False).iterrows():
        trend = "↑" if r.get("trending_up") else ("↓" if r.get("trending_up") is False else "~")
        log.debug(
            "  %-30s IC=%.4f  IC_IR=%.2f  hit_rate=%.2f  %s",
            r["factor"], r["mean_ic"], r.get("ic_ir") or 0, r.get("hit_rate") or 0, trend
        )

    # Persist to JSON
    from pathlib import Path
    from config import DATA_DIR
    out_dir = DATA_DIR / "ic_history"
    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / f"ic_{strategy}_{as_of}.json"
    ic_df.to_json(out_path, orient="records", indent=2)
    log.info("  → IC history saved to %s", out_path)

    return ic_df


def get_accuracy_summary(strategy: str) -> dict:
    """Return a high-level accuracy summary for a strategy."""
    from config import REWEIGHTING_MIN_OBSERVATIONS
    conn = get_connection()
    preds = get_predictions_with_returns(conn, strategy, "return_63d")
    conn.close()

    total = len(preds)
    filled = preds["return_63d"].notna().sum()

    return {
        "strategy":         strategy,
        "total_predictions":total,
        "filled_63d":       int(filled),
        "pct_filled":       filled / total if total > 0 else 0.0,
        "sufficient_for_reweighting": total >= REWEIGHTING_MIN_OBSERVATIONS,
        "estimated_date_sufficient": _estimate_sufficient_date(total, REWEIGHTING_MIN_OBSERVATIONS),
    }


def _estimate_sufficient_date(current: int, needed: int) -> Optional[str]:
    if current >= needed:
        return "Now"
    weeks_needed = (needed - current) / 4   # ~4 data points per week
    est = date.today() + timedelta(weeks=weeks_needed)
    return est.isoformat()
