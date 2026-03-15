"""
Horizon Ledger — Daily Automation Script
Run at 6:30 PM ET via Task Scheduler (Windows) or cron/systemd.

Steps:
  1. Fetch latest prices for all universe tickers
  2. Update technical indicators cache
  3. Fill in realized forward returns in predictions table
  4. Check for drift in all indexes
  5. Update macro indicators from FRED
  6. Log completion with counts

Usage: python scripts/run_daily.py
"""

import logging
import sys
import traceback
from datetime import date
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "data" / "daily.log"),
    ],
)
log = logging.getLogger("run_daily")


def main():
    today = date.today().isoformat()
    log.info("=" * 60)
    log.info("Horizon Ledger Daily Run — %s", today)
    log.info("=" * 60)

    # ── 1. Fetch latest prices ─────────────────────────────────────────────
    log.info("Step 1/5: Fetching latest prices...")
    try:
        from db.schema import get_connection
        from db.queries import get_universe_tickers
        from pipeline.prices import fetch_and_store_prices
        from config import BENCHMARK_TICKER

        conn = get_connection()
        tickers = get_universe_tickers(conn)
        conn.close()
        if BENCHMARK_TICKER not in tickers:
            tickers.append(BENCHMARK_TICKER)

        # Also fetch sector ETFs for sector momentum
        sector_etfs = ["XLK", "XLV", "XLF", "XLY", "XLI", "XLC", "XLP", "XLE", "XLB", "XLRE", "XLU"]
        for etf in sector_etfs:
            if etf not in tickers:
                tickers.append(etf)

        result = fetch_and_store_prices(tickers, full_history=False)
        log.info("  → Prices updated for %d tickers", len(result))
    except Exception as e:
        log.error("Price fetch failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_daily Step 1 failed", traceback.format_exc())
        except Exception:
            pass

    # ── 2. Update technical indicators ────────────────────────────────────
    log.info("Step 2/5: Updating technical indicators cache...")
    try:
        from pipeline.technicals import update_technicals_cache
        update_technicals_cache()
        log.info("  → Technical indicators updated")
    except Exception as e:
        log.error("Technicals update failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_daily Step 2 failed", traceback.format_exc())
        except Exception:
            pass

    # ── 3. Fill forward returns ───────────────────────────────────────────
    log.info("Step 3/5: Filling forward returns in predictions table...")
    try:
        from reweighting.tracker import fill_forward_returns
        counts = fill_forward_returns(as_of=today)
        total_filled = sum(counts.values())
        log.info("  → Filled %d return observations: %s", total_filled, counts)
    except Exception as e:
        log.error("Forward return fill failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_daily Step 3 failed", traceback.format_exc())
        except Exception:
            pass

    # ── 4. Check index drift ──────────────────────────────────────────────
    log.info("Step 4/5: Checking index drift...")
    try:
        from indexes.rebalancer import check_all_indexes
        drift_results = check_all_indexes()
        for idx_name, drift_df in drift_results.items():
            if not drift_df.empty:
                n_flagged = int(drift_df["exceeds_threshold"].sum())
                if n_flagged > 0:
                    log.warning("  ⚠️  %s: %d positions exceed drift threshold", idx_name, n_flagged)
                else:
                    log.info("  ✅ %s: all positions within tolerance", idx_name)
    except Exception as e:
        log.error("Drift check failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_daily Step 4 failed", traceback.format_exc())
        except Exception:
            pass

    # ── 5. Update macro indicators ────────────────────────────────────────
    log.info("Step 5/9: Updating FRED macro indicators (incl. Sahm Rule, TIPS)...")
    try:
        from pipeline.macro import update_macro_data
        update_macro_data(years=2)
        log.info("  → Macro data updated (DGS10, DGS2, VIX, HY spread, CPI, SAHMREALTIME, T10YIE)")
    except Exception as e:
        log.error("Macro update failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_daily Step 5 failed", traceback.format_exc())
        except Exception:
            pass

    # ── 6. Compute pct_above_200sma ───────────────────────────────────────
    # Must run AFTER technicals (step 2) so parquet cache is fresh
    log.info("Step 6/9: Computing %% of universe above 200-day SMA...")
    pct_above = None
    try:
        from pipeline.macro import compute_pct_above_200sma
        pct_above = compute_pct_above_200sma()
        if pct_above is not None:
            log.info("  → %%.above_200sma = %.1f%%", pct_above)
        else:
            log.warning("  → Could not compute pct_above_200sma (insufficient data)")
    except Exception as e:
        log.error("pct_above_200sma failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_daily Step 6 failed", traceback.format_exc())
        except Exception:
            pass

    # ── 7. Update market digest ───────────────────────────────────────────
    log.info("Step 7/9: Updating market digest and health score...")
    try:
        from pipeline.market_health import update_market_digest
        digest = update_market_digest(as_of=today)
        log.info(
            "  → Digest updated: score=%d (%s), regime=%s",
            digest.get("market_health_score", 0),
            digest.get("market_health_label", "unknown"),
            digest.get("regime", "unknown"),
        )
    except Exception as e:
        log.error("Market digest update failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_daily Step 7 failed", traceback.format_exc())
        except Exception:
            pass

    # ── 8. Update paper portfolio current values ──────────────────────────
    log.info("Step 8/9: Syncing paper portfolio values...")
    try:
        from paper_trading.engine import PORTFOLIO_ID_MAP, get_portfolio_value
        for strategy, pid in PORTFOLIO_ID_MAP.items():
            val = get_portfolio_value(pid)
            log.info("  → %s portfolio value: $%.2f", strategy, val)
    except Exception as e:
        log.error("Paper portfolio sync failed: %s", e)
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_daily Step 8 failed", traceback.format_exc())
        except Exception:
            pass

    # ── 9. Log completion ─────────────────────────────────────────────────
    log.info("Step 9/9: Daily run complete.")
    log.info("Daily run complete — %s", today)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            from alerts.email_alerts import send_alert
            send_alert("[Horizon Ledger] run_daily CRASHED", traceback.format_exc())
        except Exception:
            pass
        raise
