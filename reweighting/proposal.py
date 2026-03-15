"""
Horizon Ledger — Reweighting Proposal Generator
Creates a structured, human-readable reweighting proposal and saves to DB.
"""

import json
import logging
from datetime import date
from typing import Optional

import pandas as pd

from config import STRATEGY_WEIGHTS
from db.schema import get_connection
from db.queries import get_active_weights, get_latest_weight_version

log = logging.getLogger(__name__)


def generate_proposal(
    strategy: str,
    horizon: str = "return_63d",
    save_to_db: bool = True,
) -> dict:
    """
    Run the full reweighting optimization pipeline and generate a proposal.

    Steps:
      1. Get current weights
      2. Run ensemble optimizer (IC + ElasticNet)
      3. Enrich with macro regime context
      4. Save proposal to reweighting_proposals table
      5. Return full proposal dict
    """
    from reweighting.ensemble import compute_ensemble_weights

    conn = get_connection()
    current_weights = get_active_weights(conn, strategy)
    current_version = get_latest_weight_version(conn, strategy)
    conn.close()

    log.info("Generating reweighting proposal for strategy '%s'...", strategy)
    ensemble = compute_ensemble_weights(strategy, current_weights, horizon)

    if not ensemble.get("sufficient_data"):
        return {
            "strategy":  strategy,
            "status":    "insufficient_data",
            "message":   ensemble.get("message", "Insufficient data"),
            "proposal_date": date.today().isoformat(),
        }

    # Macro regime context
    regime_context = _get_regime_context()

    proposal = {
        "strategy":               strategy,
        "proposal_date":          date.today().isoformat(),
        "current_weights":        current_weights,
        "current_version":        current_version,
        "proposed_weights":       ensemble["proposed_weights"],
        "ic_weights":             ensemble["ic_weights"],
        "elastic_net_weights":    ensemble["elastic_net_weights"],
        "ensemble_weights":       ensemble["proposed_weights"],
        "confidence_intervals":   ensemble["confidence_intervals"],
        "ic_summary":             _serialize_ic_stats(ensemble.get("ic_stats", pd.DataFrame())),
        "walk_forward_efficiency":ensemble.get("wfe"),
        "estimated_turnover_impact": ensemble.get("estimated_turnover_impact"),
        "flagged_negative_ic":    ensemble.get("flagged_negative", []),
        "flagged_low_ic":         ensemble.get("flagged_low", []),
        "dropped_by_elastic_net": ensemble.get("dropped_factors", []),
        "low_t_stat_factors":     ensemble.get("low_t_factors", []),
        "t_stats":                ensemble.get("t_stats", {}),
        "en_alpha":               ensemble.get("en_alpha"),
        "en_l1_ratio":            ensemble.get("en_l1_ratio"),
        "en_oos_r2":              ensemble.get("en_oos_r2"),
        "max_change":             ensemble.get("max_change"),
        "auto_approvable":        ensemble.get("auto_approvable"),
        "recommendation":         ensemble.get("recommendation"),
        "recommendation_reason":  ensemble.get("recommendation_reason"),
        "n_observations":         ensemble.get("n_observations"),
        "macro_regime":           regime_context,
        "status":                 "pending",
    }

    if save_to_db:
        _save_proposal(proposal)
        log.info("Proposal saved to DB for strategy '%s'", strategy)

    return proposal


def _save_proposal(proposal: dict) -> None:
    """Insert the proposal into the reweighting_proposals table."""
    conn = get_connection()
    with conn:
        conn.execute(
            """INSERT INTO reweighting_proposals (
                strategy, proposal_date, current_weights, proposed_weights,
                ic_weights, elastic_net_weights, ensemble_weights,
                confidence_intervals, ic_summary, walk_forward_efficiency,
                estimated_turnover_impact, status, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                proposal["strategy"],
                proposal["proposal_date"],
                json.dumps(proposal["current_weights"]),
                json.dumps(proposal["proposed_weights"]),
                json.dumps(proposal["ic_weights"]),
                json.dumps(proposal["elastic_net_weights"]),
                json.dumps(proposal["ensemble_weights"]),
                json.dumps(proposal["confidence_intervals"]),
                json.dumps(proposal["ic_summary"]),
                proposal.get("walk_forward_efficiency"),
                proposal.get("estimated_turnover_impact"),
                "auto_approved" if proposal.get("auto_approvable") else "pending",
                proposal.get("recommendation_reason"),
            ),
        )
        conn.commit()

        # Auto-approve if eligible
        if proposal.get("auto_approvable"):
            _auto_approve(conn, proposal)

    conn.close()


def _auto_approve(conn, proposal: dict) -> None:
    """
    Auto-approve a proposal if all guardrails pass.
    Logs reason for transparency.
    """
    from db.queries import get_latest_weight_version
    today = date.today().isoformat()
    strategy = proposal["strategy"]
    new_weights = proposal["proposed_weights"]
    old_version = get_latest_weight_version(conn, strategy) or "v0"
    # Increment version
    parts = old_version.split("-")
    new_version = f"{today[:7]}-auto"
    conn.execute(
        """INSERT INTO weight_versions
           (strategy, version_id, weights, created_date, approved_by, notes)
           VALUES (?,?,?,?,?,?)""",
        (
            strategy, new_version, json.dumps(new_weights), today,
            "auto_approval",
            f"Auto-approved: all changes < 2%, CIs positive. Prev: {old_version}",
        ),
    )
    conn.execute(
        """UPDATE reweighting_proposals SET status='auto_approved', approved_date=?,
           approved_weights=? WHERE strategy=? AND status='pending'
           ORDER BY id DESC LIMIT 1""",
        (today, json.dumps(new_weights), strategy),
    )
    conn.commit()
    log.info("✅ Auto-approved reweighting for '%s' (version %s)", strategy, new_version)


def _get_regime_context() -> dict:
    """Get current HMM regime for inclusion in proposal."""
    try:
        from db.queries import get_current_regime
        from db.schema import get_connection as gc
        conn = gc()
        regime = get_current_regime(conn)
        conn.close()
        if regime:
            return dict(regime)
    except Exception:
        pass
    return {"regime": "unknown"}


def _serialize_ic_stats(ic_df: pd.DataFrame) -> dict:
    """Convert IC stats DataFrame to a JSON-serializable dict."""
    if ic_df is None or ic_df.empty:
        return {}
    return ic_df.to_dict(orient="records")


def get_pending_proposals(strategy: Optional[str] = None) -> list[dict]:
    """Return all pending reweighting proposals."""
    conn = get_connection()
    query = "SELECT * FROM reweighting_proposals WHERE status='pending'"
    params = []
    if strategy:
        query += " AND strategy=?"
        params.append(strategy)
    query += " ORDER BY proposal_date DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()

    proposals = []
    for r in rows:
        p = dict(r)
        for field in ["current_weights", "proposed_weights", "ic_weights",
                       "elastic_net_weights", "ensemble_weights",
                       "confidence_intervals", "ic_summary"]:
            if p.get(field):
                try:
                    p[field] = json.loads(p[field])
                except Exception:
                    pass
        proposals.append(p)
    return proposals


def approve_proposal(proposal_id: int, approved_weights: dict, approved_by: str = "human") -> None:
    """Approve a reweighting proposal and update weight_versions."""
    today = date.today().isoformat()
    conn = get_connection()

    row = conn.execute(
        "SELECT * FROM reweighting_proposals WHERE id=?", (proposal_id,)
    ).fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"Proposal {proposal_id} not found")

    strategy = row["strategy"]
    old_version = get_latest_weight_version(conn, strategy) or "v0"
    new_version = f"{today[:7]}-human"

    with conn:
        conn.execute(
            """UPDATE reweighting_proposals
               SET status='approved', approved_date=?, approved_weights=?
               WHERE id=?""",
            (today, json.dumps(approved_weights), proposal_id),
        )
        conn.execute(
            """INSERT INTO weight_versions
               (strategy, version_id, weights, created_date, approved_by, notes)
               VALUES (?,?,?,?,?,?)""",
            (strategy, new_version, json.dumps(approved_weights), today,
             approved_by, f"Human-approved. Previous version: {old_version}"),
        )
        conn.commit()
    conn.close()
    log.info("✅ Proposal %d approved for '%s' (version %s)", proposal_id, strategy, new_version)


def reject_proposal(proposal_id: int, reason: str = "") -> None:
    """Reject a reweighting proposal."""
    conn = get_connection()
    with conn:
        conn.execute(
            """UPDATE reweighting_proposals
               SET status='rejected', notes=?
               WHERE id=?""",
            (reason, proposal_id),
        )
        conn.commit()
    conn.close()
    log.info("❌ Proposal %d rejected: %s", proposal_id, reason)


def generate_all_proposals() -> list[dict]:
    """Generate proposals for all four strategies."""
    strategies = ["long_term", "dividend", "turnaround", "swing"]
    horizon_map = {
        "long_term":  "return_63d",
        "dividend":   "return_63d",
        "turnaround": "return_63d",
        "swing":      "return_21d",
    }
    results = []
    for strategy in strategies:
        horizon = horizon_map[strategy]
        try:
            proposal = generate_proposal(strategy, horizon)
            results.append(proposal)
            log.info(
                "Proposal for %s: %s (%s)",
                strategy,
                proposal.get("recommendation", "N/A"),
                proposal.get("status", "N/A"),
            )
        except Exception as e:
            log.error("Proposal generation failed for %s: %s", strategy, e)
    return results
