"""
Horizon Ledger — Weekly Automation Script
Run Saturday morning.

Steps:
  1. Refresh stock universe
  2. Update SEC EDGAR fundamentals (delta update)
  3. Run all four scoring engines across full universe
  4. Fill forward returns in predictions table
  5. Check if quarterly rebalancing is due
  6. Update HMM regime detector
  7. Compute cross-strategy consensus top-25
  8. Sync paper portfolios to latest index composition
  9. Generate weekly performance report
 10. Export public data for GitHub Pages
 11. Generate and send newsletter (first Sunday of month only)

Usage: python scripts/run_weekly.py
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
        logging.FileHandler(Path(__file__).parent.parent / "data" / "weekly.log"),
    ],
)
log = logging.getLogger("run_weekly")


def main():
    today = date.today().isoformat()
    log.info("=" * 60)
    log.info("Horizon Ledger Weekly Run — %s", today)
    log.info("=" * 60)

    # ── 0. Ensure DB schema ───────────────────────────────────────────────
    from db.schema import init_db, seed_initial_weights, get_connection
    init_db()
    conn = get_connection()
    seed_initial_weights(conn)
    conn.close()

    # ── 1. Refresh universe ───────────────────────────────────────────────
    log.info("Step 1/11: Refreshing stock universe...")
    try:
        from pipeline.universe import build_universe, refresh_cik_mapping
        build_universe(apply_filters=False)   # Fast refresh without volume filter
        refresh_cik_mapping()
        log.info("  → Universe refreshed")
    except Exception as e:
        log.error("Universe refresh failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_weekly Step 1 failed", traceback.format_exc())
        except Exception:
            pass

    # ── 2. Update EDGAR fundamentals ──────────────────────────────────────
    log.info("Step 2/11: Updating SEC EDGAR fundamentals...")
    try:
        from pipeline.edgar_bulk import update_fundamentals_from_edgar
        update_fundamentals_from_edgar()
        log.info("  → EDGAR fundamentals updated")
    except Exception as e:
        log.error("EDGAR update failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_weekly Step 2 failed", traceback.format_exc())
        except Exception:
            pass
        # Fallback to yfinance
        log.info("  → Trying yfinance fundamentals fallback...")
        try:
            from pipeline.fundamentals import update_fundamentals_for_universe
            update_fundamentals_for_universe()
            log.info("  → yfinance fundamentals updated")
        except Exception as e2:
            log.error("  yfinance fallback also failed: %s", e2)

    # ── 3. Run all scoring engines ────────────────────────────────────────
    log.info("Step 3/11: Running scoring engines...")
    try:
        from scoring.long_term  import score_universe as lt
        from scoring.dividend   import score_universe as dv
        from scoring.turnaround import score_universe as ta
        from scoring.swing      import score_universe as sw
        from scoring.conservative import score_universe as cv
        from scoring.aggressive   import score_universe as ag

        log.info("  Long-term scoring...")
        lt_df = lt(as_of=today, persist=True)
        log.info("  → %d stocks scored (long_term)", len(lt_df))

        log.info("  Dividend scoring...")
        dv_df = dv(as_of=today, persist=True)
        log.info("  → %d stocks scored (dividend)", len(dv_df))

        log.info("  Turnaround scoring...")
        ta_df = ta(as_of=today, persist=True)
        log.info("  → %d stocks scored (turnaround)", len(ta_df))

        log.info("  Swing scoring...")
        sw_df = sw(as_of=today, persist=True)
        log.info("  → %d stocks scored (swing)", len(sw_df))

        log.info("  Conservative scoring...")
        cv_df = cv(as_of=today, persist=True)
        log.info("  → %d stocks scored (conservative)", len(cv_df))

        log.info("  Aggressive scoring...")
        ag_df = ag(as_of=today, persist=True)
        log.info("  → %d stocks scored (aggressive)", len(ag_df))

    except Exception as e:
        log.error("Scoring failed: %s", e, exc_info=True)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_weekly Step 3 failed", traceback.format_exc())
        except Exception:
            pass

    # ── 4. Fill forward returns ───────────────────────────────────────────
    log.info("Step 4/11: Filling prediction forward returns...")
    try:
        from reweighting.tracker import fill_forward_returns
        counts = fill_forward_returns(as_of=today)
        log.info("  → Filled returns: %s", counts)
    except Exception as e:
        log.error("Forward return fill failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_weekly Step 4 failed", traceback.format_exc())
        except Exception:
            pass

    # ── 4b. Update IC statistics (runs after forward returns are filled) ─────
    log.info("Step 4b: Updating IC statistics per strategy...")
    all_strategies = ["long_term", "dividend", "turnaround", "swing", "conservative", "aggressive"]
    for _strat in all_strategies:
        try:
            from reweighting.tracker import store_ic_statistics
            store_ic_statistics(_strat, as_of=today)
        except Exception as _e:
            log.warning("  IC statistics for '%s' skipped: %s", _strat, _e)

    # ── 5. Check quarterly rebalancing + reconstitute aggressive (monthly) ──
    log.info("Step 5/11: Checking rebalancing schedule...")
    try:
        from indexes.rebalancer import is_quarterly_rebalance_due
        if is_quarterly_rebalance_due():
            log.info("  ⏰ Quarterly rebalance due — run run_quarterly.py")
        else:
            log.info("  → Not a quarterly rebalance week")
    except Exception as e:
        log.error("Rebalance check failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_weekly Step 5 failed", traceback.format_exc())
        except Exception:
            pass

    # Aggressive index reconstitutes monthly — run every first Saturday of the month
    import calendar as _cal2
    _td = date.today()
    _first_sat = min(d for d in range(1, 8) if _cal2.weekday(_td.year, _td.month, d) == _cal2.SATURDAY)
    if _td.day == _first_sat:
        log.info("Step 5b: Monthly aggressive index reconstitution...")
        try:
            from indexes.builder import reconstitute_index
            result = reconstitute_index("aggressive", as_of=today, dry_run=False)
            log.info(
                "  → Aggressive: +%d additions, -%d removals",
                len(result["adds"]), len(result["removes"])
            )
        except Exception as e:
            log.error("Aggressive reconstitution failed: %s", e)
            try:
                from alerts.email_alerts import send_alert
                send_alert("[Horizon Ledger] run_weekly Step 5b (aggressive rebalance) failed", traceback.format_exc())
            except Exception:
                pass
    else:
        log.info("Step 5b: Not first Saturday — aggressive rebalance skipped.")

    # ── 6. Update HMM regime ──────────────────────────────────────────────
    log.info("Step 6/11: Updating HMM regime detector...")
    try:
        from regime.hmm_detector import update_regime_history
        regime = update_regime_history(retrain=False)
        log.info("  → Regime: %s", regime.get("regime", "unknown"))
    except Exception as e:
        log.error("Regime update failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_weekly Step 6 failed", traceback.format_exc())
        except Exception:
            pass

    # ── 7. Compute cross-strategy consensus top-25 ────────────────────────
    # Must run AFTER scoring (step 3) so all strategy scores are current.
    log.info("Step 7/11: Computing cross-strategy consensus top-25...")
    try:
        from scoring.consensus import store_consensus_in_digest
        store_consensus_in_digest(as_of=today)
        log.info("  → Consensus top-25 computed and stored in market digest")
    except Exception as e:
        log.error("Consensus top-25 failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_weekly Step 7 failed", traceback.format_exc())
        except Exception:
            pass

    # ── 8. Sync paper portfolios to latest index composition ──────────────
    # Mirrors any index reconstitution changes into the paper portfolio.
    log.info("Step 8/11: Syncing paper portfolios to index composition...")
    try:
        from paper_trading.engine import sync_all_portfolios
        sync_all_portfolios(as_of=today)
        log.info("  → All paper portfolios synced")
    except Exception as e:
        log.error("Paper portfolio sync failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_weekly Step 8 failed", traceback.format_exc())
        except Exception:
            pass

    # ── 9. Weekly performance report ──────────────────────────────────────
    log.info("Step 9/11: Generating weekly performance report...")
    try:
        _generate_weekly_report(today)
    except Exception as e:
        log.error("Performance report failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_weekly Step 9 failed", traceback.format_exc())
        except Exception:
            pass

    # ── 10. Export public data for GitHub Pages + auto-push ──────────────
    log.info("Step 10/11: Exporting public data for GitHub Pages...")
    try:
        from scripts.export_public_data import export_all
        export_all()
        log.info("  → Public data exported to docs/data/")
    except Exception as e:
        log.error("Public data export failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_weekly Step 10 (public data export) failed", traceback.format_exc())
        except Exception:
            pass

    # ── 10b. Git push docs/data/ to GitHub Pages ──────────────────────────
    log.info("Step 10b: Pushing public data to GitHub Pages...")
    try:
        import subprocess
        from config import BASE_DIR
        _repo = str(BASE_DIR)

        def _git(*args):
            return subprocess.run(
                ["git"] + list(args),
                cwd=_repo,
                capture_output=True,
                text=True,
            )

        # Stage only the data folder (never commits source code)
        _git("add", "docs/data/")

        # Check if there's anything new to commit
        status = _git("status", "--porcelain", "docs/data/")
        if status.stdout.strip():
            _git("commit", "-m", f"Weekly data update {today}")
            push_result = _git("push")
            if push_result.returncode == 0:
                log.info("  → GitHub Pages data pushed successfully")
            else:
                log.error("  → Git push failed: %s", push_result.stderr.strip())
        else:
            log.info("  → No data changes to push")
    except Exception as e:
        log.error("GitHub Pages push failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_weekly Step 10b (git push) failed", traceback.format_exc())
        except Exception:
            pass

    # ── 11. Newsletter: generate + email (first Sunday of month only) ─────
    import calendar as _cal
    today_obj = date.today()
    # First Sunday of the month
    first_sun = min(
        d for d in range(1, 8)
        if _cal.weekday(today_obj.year, today_obj.month, d) == _cal.SUNDAY
    )
    is_newsletter_day = (today_obj.weekday() == _cal.SUNDAY and today_obj.day == first_sun)

    if is_newsletter_day:
        log.info("Step 11/11: Generating and sending monthly newsletter...")
        try:
            from newsletter.generator import generate_newsletter, get_next_issue_number
            from alerts.email_alerts import send_newsletter
            from db.schema import get_connection as _gc
            _conn = _gc()
            issue_num = get_next_issue_number(_conn)
            _conn.close()
            pdf_path = generate_newsletter(today, issue_num)
            log.info("  → Newsletter PDF generated: %s", pdf_path)
            # Send to mailing list
            from alerts.email_alerts import get_recipients
            recipients = get_recipients()
            if recipients:
                ok = send_newsletter(pdf_path, recipients, today, issue_num)
                log.info("  → Newsletter sent to %d recipients: %s", len(recipients), ok)
            else:
                log.info("  → No newsletter recipients configured. PDF saved at %s", pdf_path)
        except Exception as e:
            log.error("Newsletter generation/send failed: %s", e)
            try:
                from alerts.email_alerts import send_alert
                send_alert("[Horizon Ledger] run_weekly Step 11 (newsletter) failed", traceback.format_exc())
            except Exception:
                pass
    else:
        log.info("Step 11/11: Not newsletter day (first Sunday of month). Skipping.")

    log.info("Weekly run complete — %s", today)


def _generate_weekly_report(today: str) -> None:
    """Write a markdown weekly performance summary."""
    from indexes.performance import get_all_metrics
    from indexes.builder import INDEX_NAMES

    report_path = Path(__file__).parent.parent / "data" / f"weekly_report_{today}.md"
    lines = [f"# Horizon Ledger Weekly Report — {today}\n"]
    lines.append("⚠️ NOT FINANCIAL ADVICE\n\n")

    for strategy, index_name in INDEX_NAMES.items():
        try:
            m = get_all_metrics(index_name)
            lines.append(f"## {strategy.upper().replace('_', ' ')}\n")
            lines.append(f"- Annual Return: {m.get('annual_return', 0):+.1%}\n")
            lines.append(f"- vs Benchmark:  {m.get('annual_return', 0) - m.get('benchmark_return', 0):+.1%}\n")
            lines.append(f"- Sharpe Ratio:  {m.get('sharpe_ratio', 0):.2f}\n")
            lines.append(f"- Max Drawdown:  {m.get('max_drawdown', 0):.1%}\n")
            lines.append(f"- Alpha:         {m.get('alpha', 0):+.1%}\n\n")
        except Exception as e:
            lines.append(f"## {strategy} — Error: {e}\n\n")

    # Paper portfolio summary
    try:
        from paper_trading.engine import PORTFOLIO_ID_MAP, get_portfolio_value, get_performance_metrics
        lines.append("## PAPER PORTFOLIOS\n")
        for strategy, pid in PORTFOLIO_ID_MAP.items():
            val = get_portfolio_value(pid)
            try:
                metrics = get_performance_metrics(pid)
                combined = metrics.get("combined", {})
                lines.append(
                    f"- {strategy}: ${val:,.2f}  |  "
                    f"Return: {combined.get('total_return', 0):+.1%}  |  "
                    f"Sharpe: {combined.get('sharpe_ratio', 0):.2f}\n"
                )
            except Exception:
                lines.append(f"- {strategy}: ${val:,.2f}\n")
        lines.append("\n")
    except Exception as e:
        lines.append(f"## PAPER PORTFOLIOS — Error: {e}\n\n")

    report_path.write_text("".join(lines), encoding="utf-8")
    log.info("  → Weekly report written to %s", report_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_weekly CRASHED", traceback.format_exc())
        except Exception:
            pass
        raise
