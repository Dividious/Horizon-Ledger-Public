"""
Horizon Ledger — Quarterly Automation Script
Run first Saturday of Jan/Apr/Jul/Oct.

Steps:
  1. Full universe reconstitution (re-score, re-rank, apply banding)
  2. Generate rebalancing proposals for all four indexes
  3. Run reweighting optimization if ≥60 monthly predictions available
  4. Retrain HMM regime model
  5. Refresh Shiller CAPE historical data
  6. Mirror index changes to paper portfolios (rebalance)
  7. Print summary

Usage: python scripts/run_quarterly.py
"""

import logging
import sys
import traceback
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "data" / "quarterly.log"),
    ],
)
log = logging.getLogger("run_quarterly")


def main():
    today = date.today().isoformat()
    log.info("=" * 60)
    log.info("Horizon Ledger Quarterly Run — %s", today)
    log.info("=" * 60)

    from config import QUARTERLY_MONTHS
    if date.today().month not in QUARTERLY_MONTHS:
        log.warning("Month %d is not a quarterly month. Running anyway...", date.today().month)

    # ── 1. Full universe reconstitution ───────────────────────────────────
    log.info("Step 1/6: Full index reconstitution for all strategies...")
    strategies = ["long_term", "dividend", "turnaround", "swing", "conservative"]
    recon_results = {}
    for strategy in strategies:
        try:
            from indexes.builder import reconstitute_index
            result = reconstitute_index(strategy, as_of=today, dry_run=False)
            recon_results[strategy] = result
            log.info(
                "  %s: +%d additions, -%d removals, ~%d weight changes",
                strategy,
                len(result["adds"]),
                len(result["removes"]),
                len(result["weight_changes"]),
            )
        except Exception as e:
            log.error("  Reconstitution failed for %s: %s", strategy, e)
            try:
                from alerts.email_alerts import send_alert
                send_alert("[Horizon Ledger] run_quarterly Step 1 (reconstitution) failed", traceback.format_exc())
            except Exception:
                pass

    # ── 2. Generate rebalancing proposals ─────────────────────────────────
    log.info("Step 2/6: Generating rebalancing proposals...")
    from indexes.rebalancer import generate_rebalancing_proposal
    from indexes.builder import INDEX_NAMES
    for strategy, index_name in INDEX_NAMES.items():
        try:
            proposal = generate_rebalancing_proposal(index_name, as_of=today)
            log.info(
                "  %s: needs_rebalancing=%s, max_drift=%.2f%%",
                index_name,
                proposal["needs_rebalancing"],
                proposal.get("max_drift", 0) * 100,
            )
        except Exception as e:
            log.error("  Rebalancing proposal failed for %s: %s", index_name, e)
            try:
                from alerts.email_alerts import send_alert
                send_alert("[Horizon Ledger] run_quarterly Step 2 (rebalancing proposals) failed", traceback.format_exc())
            except Exception:
                pass

    # ── 3. Run reweighting optimization ───────────────────────────────────
    log.info("Step 3/6: Running reweighting optimization...")
    from reweighting.tracker import get_accuracy_summary
    from reweighting.proposal import generate_all_proposals
    any_proposal = False
    for strategy in strategies:
        accuracy = get_accuracy_summary(strategy)
        if accuracy["sufficient_for_reweighting"]:
            any_proposal = True
        else:
            log.info(
                "  %s: Insufficient data for reweighting (%d observations, need %d). Est: %s",
                strategy,
                accuracy["total_predictions"],
                60,
                accuracy["estimated_date_sufficient"],
            )

    if any_proposal:
        proposals = generate_all_proposals()
        for p in proposals:
            log.info(
                "  Proposal for %s: %s — %s",
                p.get("strategy"),
                p.get("recommendation", "N/A"),
                p.get("status", "N/A"),
            )
        log.info("  → Reweighting proposals saved. Review in dashboard > Reweighting tab.")
    else:
        log.info("  → No strategies have sufficient data yet. Continue collecting predictions.")

    # ── 4. Retrain HMM ────────────────────────────────────────────────────
    log.info("Step 4/6: Retraining HMM regime model...")
    try:
        from regime.hmm_detector import update_regime_history
        regime = update_regime_history(retrain=True)
        log.info(
            "  → HMM retrained. Current regime: %s (%.0f%% confidence)",
            regime.get("regime", "unknown"),
            max(
                regime.get("prob_bear", 0),
                regime.get("prob_neutral", 0),
                regime.get("prob_bull", 0),
            ) * 100,
        )
    except Exception as e:
        log.error("  HMM retrain failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_quarterly Step 4 (HMM retrain) failed", traceback.format_exc())
        except Exception:
            pass

    # ── 5. Refresh Shiller CAPE data ──────────────────────────────────────
    # Downloads the latest Yale Shiller spreadsheet and updates the macro_data
    # table with a full CAPE history refresh.
    log.info("Step 5/6: Refreshing Shiller CAPE historical data...")
    try:
        from pipeline.macro import fetch_cape_data, update_cape_in_macro_db
        cape_df = fetch_cape_data()
        if cape_df is not None and not cape_df.empty:
            update_cape_in_macro_db(cape_df)
            log.info(
                "  → CAPE data refreshed: %d monthly observations (latest: %s)",
                len(cape_df),
                cape_df.index.max() if hasattr(cape_df.index, "max") else "N/A",
            )
        else:
            log.warning("  → CAPE fetch returned no data (Yale server may be down)")
    except Exception as e:
        log.error("  CAPE refresh failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_quarterly Step 5 (CAPE refresh) failed", traceback.format_exc())
        except Exception:
            pass

    # ── 6. Mirror index reconstitution to paper portfolios ────────────────
    # Rebalances each paper portfolio to match the freshly reconstituted index.
    # This is the primary quarterly rebalance for the simulation engine.
    log.info("Step 6/6: Mirroring index reconstitution to paper portfolios...")
    try:
        from paper_trading.engine import sync_all_portfolios
        sync_all_portfolios(as_of=today)
        log.info("  → All paper portfolios rebalanced to match current index composition")
    except Exception as e:
        log.error("  Paper portfolio rebalancing failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_quarterly Step 6 (paper portfolio rebalance) failed", traceback.format_exc())
        except Exception:
            pass

    # ── Summary ────────────────────────────────────────────────────────────
    total_adds    = sum(len(r.get("adds", [])) for r in recon_results.values())
    total_removes = sum(len(r.get("removes", [])) for r in recon_results.values())
    log.info("=" * 60)
    log.info("Quarterly Summary:")
    log.info("  %d new additions, %d removals across %d indexes", total_adds, total_removes, len(strategies))
    if any_proposal:
        log.info("  Reweighting proposal generated — see dashboard for review.")
    else:
        log.info("  No reweighting proposals yet (insufficient history).")
    log.info("  CAPE data refreshed from Yale Shiller spreadsheet.")
    log.info("  Paper portfolios rebalanced to match latest index composition.")
    log.info("Quarterly run complete — %s", today)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_quarterly CRASHED", traceback.format_exc())
        except Exception:
            pass
        raise
