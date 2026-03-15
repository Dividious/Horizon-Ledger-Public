"""
Horizon Ledger — Historical Backfill Script
One-time script to populate initial 5-year history.

⚠️  SURVIVORSHIP BIAS WARNING:
    This script downloads data for CURRENT universe members only.
    Historical backtests produced from this data are subject to survivorship bias.
    Forward-looking predictions (from the date this script first runs) are
    universe-bias-free.

Steps:
  1. Initialize database
  2. Build stock universe (S&P 500 + Russell 1000)
  3. Refresh CIK mapping from SEC
  4. Backfill 5 years of daily prices
  5. Download EDGAR fundamentals (last 8 quarters)
  6. Download FRED macro data (last 12 years)
  7. Compute historical technical indicators
  8. Run scoring on latest data
  9. Compute initial regime estimate

Usage: python scripts/backfill_history.py
"""

import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("backfill")

SURVIVORSHIP_WARNING = """
╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  SURVIVORSHIP BIAS WARNING                                   ║
║                                                                   ║
║  This backfill downloads data for current universe members ONLY.  ║
║  Stocks that were delisted, went bankrupt, or were removed from   ║
║  indices between 2019 and today are NOT included.                 ║
║                                                                   ║
║  Historical analysis from this data will be biased upward.       ║
║  This is a known limitation of all live-universe backfills.       ║
║                                                                   ║
║  Forward predictions (from today onward) are NOT subject to       ║
║  this bias — all stocks in the universe at signal time are scored. ║
╚══════════════════════════════════════════════════════════════════╝
"""


def main():
    today = date.today().isoformat()
    print(SURVIVORSHIP_WARNING)
    log.info("=" * 60)
    log.info("Horizon Ledger Historical Backfill — %s", today)
    log.info("=" * 60)

    # ── 1. Initialize database ────────────────────────────────────────────
    log.info("Step 1/9: Initializing database...")
    from db.schema import init_db, seed_initial_weights, get_connection
    init_db()
    conn = get_connection()
    seed_initial_weights(conn)
    conn.close()
    log.info("  → Database initialized")

    # ── 2. Build universe ─────────────────────────────────────────────────
    log.info("Step 2/9: Building stock universe (S&P 500 + Russell 1000)...")
    try:
        from pipeline.universe import build_universe
        build_universe(apply_filters=True)
        conn = get_connection()
        n = conn.execute("SELECT COUNT(*) FROM stocks WHERE is_active=1").fetchone()[0]
        conn.close()
        log.info("  → %d active stocks in universe", n)
    except Exception as e:
        log.error("Universe build failed: %s", e, exc_info=True)
        return

    # ── 3. Refresh CIK mapping ────────────────────────────────────────────
    log.info("Step 3/9: Refreshing SEC CIK mapping...")
    try:
        from pipeline.universe import refresh_cik_mapping
        refresh_cik_mapping()
        log.info("  → CIK mapping refreshed")
    except Exception as e:
        log.error("CIK mapping failed: %s", e)

    # ── 4. Backfill prices ────────────────────────────────────────────────
    log.info("Step 4/9: Backfilling 5 years of daily prices (this may take a while)...")
    try:
        from pipeline.prices import backfill_prices
        backfill_prices()
        log.info("  → Prices backfilled")
    except Exception as e:
        log.error("Price backfill failed: %s", e, exc_info=True)

    # ── 5. Download EDGAR fundamentals ───────────────────────────────────
    log.info("Step 5/9: Downloading SEC EDGAR fundamentals...")
    try:
        from pipeline.edgar_bulk import update_fundamentals_from_edgar
        update_fundamentals_from_edgar()
        conn = get_connection()
        n_fund = conn.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0]
        conn.close()
        log.info("  → %d fundamental rows stored", n_fund)
    except Exception as e:
        log.error("EDGAR download failed: %s — trying yfinance fallback", e)
        try:
            from pipeline.fundamentals import update_fundamentals_for_universe
            update_fundamentals_for_universe()
            log.info("  → yfinance fundamentals downloaded as fallback")
        except Exception as e2:
            log.error("yfinance fallback also failed: %s", e2)

    # ── 6. FRED macro data ────────────────────────────────────────────────
    log.info("Step 6/9: Downloading FRED macro data (12 years)...")
    try:
        from pipeline.macro import update_macro_data
        update_macro_data(years=12)
        log.info("  → Macro data downloaded")
    except Exception as e:
        log.error("FRED download failed: %s", e)

    # ── 7. Technical indicators ───────────────────────────────────────────
    log.info("Step 7/9: Computing technical indicator cache...")
    try:
        from pipeline.technicals import update_technicals_cache
        update_technicals_cache()
        log.info("  → Technical indicators computed")
    except Exception as e:
        log.error("Technicals failed: %s", e)

    # ── 8. Initial scoring ────────────────────────────────────────────────
    log.info("Step 8/9: Running initial scoring (today = %s)...", today)
    try:
        from scoring.long_term  import score_universe as lt
        from scoring.dividend   import score_universe as dv
        from scoring.turnaround import score_universe as ta
        from scoring.swing      import score_universe as sw
        lt(as_of=today, persist=True)
        dv(as_of=today, persist=True)
        ta(as_of=today, persist=True)
        sw(as_of=today, persist=True)
        log.info("  → Initial scoring complete")
    except Exception as e:
        log.error("Scoring failed: %s", e, exc_info=True)

    # ── 9. Initial regime estimate ────────────────────────────────────────
    log.info("Step 9/9: Computing initial market regime...")
    try:
        from regime.hmm_detector import update_regime_history
        regime = update_regime_history(retrain=True)
        log.info(
            "  → Regime: %s (bear=%.0f%%, neutral=%.0f%%, bull=%.0f%%)",
            regime.get("regime", "unknown"),
            regime.get("prob_bear", 0) * 100,
            regime.get("prob_neutral", 0) * 100,
            regime.get("prob_bull", 0) * 100,
        )
    except Exception as e:
        log.error("Regime estimation failed: %s", e)

    # ── Summary ────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Backfill complete!")
    log.info("")
    log.info("Next steps:")
    log.info("  1. Launch dashboard: streamlit run main.py")
    log.info("  2. Set up daily automation: Task Scheduler or cron")
    log.info("     - Daily (6:30 PM ET): python scripts/run_daily.py")
    log.info("     - Weekly (Saturday):  python scripts/run_weekly.py")
    log.info("     - Quarterly (Jan/Apr/Jul/Oct): python scripts/run_quarterly.py")
    log.info("")
    log.info("Note: %s", SURVIVORSHIP_WARNING.strip().split("\n")[2].strip())


if __name__ == "__main__":
    main()
