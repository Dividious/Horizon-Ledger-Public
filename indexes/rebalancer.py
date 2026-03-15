"""
Horizon Ledger — Drift Monitoring and Rebalancing Proposals
Checks for weight drift weekly and generates proposals for human review.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from config import REBALANCE_DRIFT_THRESHOLD, INDEX_SIZES
from db.schema import get_connection
from db.queries import get_current_holdings, get_price_on_date

log = logging.getLogger(__name__)


def check_drift(index_name: str, as_of: Optional[str] = None) -> pd.DataFrame:
    """
    Check weight drift for all current holdings.
    Returns DataFrame with columns:
      ticker, target_weight, current_weight, drift, exceeds_threshold
    """
    if as_of is None:
        as_of = date.today().isoformat()

    conn = get_connection()
    holdings = get_current_holdings(conn, index_name)
    if holdings.empty:
        conn.close()
        return pd.DataFrame()

    # Compute current market value of each position
    current_values = {}
    for _, h in holdings.iterrows():
        price = get_price_on_date(conn, h["stock_id"], as_of) or 0.0
        entry_price = h.get("entry_price") or price or 1.0
        # Relative value: use price / entry_price as proxy for drift
        current_values[h["ticker"]] = (price / entry_price) if entry_price > 0 else 1.0

    conn.close()

    # Compute current weights (normalize price-return-adjusted values)
    target_weights = dict(zip(holdings["ticker"], holdings["target_weight"].fillna(0)))
    adjusted = {t: target_weights.get(t, 0) * v for t, v in current_values.items()}
    total_adj = sum(adjusted.values())

    rows = []
    for tkr in holdings["ticker"]:
        target_w   = target_weights.get(tkr, 0)
        current_w  = adjusted.get(tkr, 0) / total_adj if total_adj > 0 else target_w
        drift      = abs(current_w - target_w)
        rows.append({
            "ticker":           tkr,
            "target_weight":    target_w,
            "current_weight":   current_w,
            "drift":            drift,
            "exceeds_threshold": drift > REBALANCE_DRIFT_THRESHOLD,
        })

    return pd.DataFrame(rows).sort_values("drift", ascending=False)


def check_all_indexes() -> dict[str, pd.DataFrame]:
    """Check drift for all four indexes."""
    from indexes.builder import INDEX_NAMES
    results = {}
    for strategy, index_name in INDEX_NAMES.items():
        drift_df = check_drift(index_name)
        if not drift_df.empty and drift_df["exceeds_threshold"].any():
            log.warning(
                "Drift alert: %s — %d positions exceed %.0f%% threshold",
                index_name,
                drift_df["exceeds_threshold"].sum(),
                REBALANCE_DRIFT_THRESHOLD * 100,
            )
        results[index_name] = drift_df
    return results


def generate_rebalancing_proposal(
    index_name: str,
    as_of: Optional[str] = None,
) -> dict:
    """
    Generate a human-readable rebalancing proposal based on drift analysis.
    Returns dict with: index_name, drift_df, needs_rebalancing, actions
    """
    if as_of is None:
        as_of = date.today().isoformat()

    drift_df = check_drift(index_name, as_of)
    if drift_df.empty:
        return {"index_name": index_name, "needs_rebalancing": False, "drift_df": drift_df, "actions": []}

    needs_rebalancing = drift_df["exceeds_threshold"].any()
    actions = []

    if needs_rebalancing:
        for _, row in drift_df[drift_df["exceeds_threshold"]].iterrows():
            direction = "reduce" if row["current_weight"] > row["target_weight"] else "increase"
            actions.append({
                "ticker":          row["ticker"],
                "action":          direction,
                "current_weight":  row["current_weight"],
                "target_weight":   row["target_weight"],
                "drift":           row["drift"],
            })

    return {
        "index_name":       index_name,
        "as_of":            as_of,
        "needs_rebalancing": needs_rebalancing,
        "drift_df":         drift_df,
        "actions":          actions,
        "max_drift":        drift_df["drift"].max() if not drift_df.empty else 0,
        "n_flagged":        int(drift_df["exceeds_threshold"].sum()) if not drift_df.empty else 0,
    }


def is_quarterly_rebalance_due(as_of: Optional[str] = None) -> bool:
    """Return True if today is in a quarterly rebalance month (Jan/Apr/Jul/Oct)."""
    from config import QUARTERLY_MONTHS
    if as_of is None:
        today = date.today()
    else:
        today = date.fromisoformat(as_of)
    return today.month in QUARTERLY_MONTHS and today.day <= 7
