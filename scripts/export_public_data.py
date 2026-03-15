"""
Horizon Ledger — Export Public Data for GitHub Pages
Reads from SQLite DB and writes JSON files to docs/data/.
Run weekly (called from run_weekly.py Step 10).

NOT FINANCIAL ADVICE.
"""

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)


def export_all() -> None:
    """Export all public data files to docs/data/."""
    from config import DOCS_DIR
    data_dir = DOCS_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    _export_market_pulse(data_dir)
    _export_conservative_index(data_dir)
    _export_aggressive_index(data_dir)
    _export_meta(data_dir)
    log.info("export_public_data: all files written to %s", data_dir)


def _safe(v, default=None):
    """Return None-safe JSON-serializable value."""
    if v is None:
        return default
    try:
        f = float(v)
        import math
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return str(v)


def _export_market_pulse(data_dir: Path) -> None:
    """Export market_pulse.json — used on index.html and public pages."""
    from db.schema import get_connection
    conn = get_connection()
    today = date.today().isoformat()

    try:
        cur = conn.execute(
            "SELECT * FROM market_digest WHERE digest_date <= ? ORDER BY digest_date DESC LIMIT 1",
            (today,)
        )
        row = cur.fetchone()
        if row is None:
            conn.close()
            return
        d = dict(row)

        # Parse consensus JSON if present
        import json as _json
        consensus = []
        try:
            consensus = _json.loads(d.get("consensus_tickers") or "[]")
        except Exception:
            pass

        pulse = {
            "as_of": d.get("digest_date", today),
            "market_health_score": _safe(d.get("market_health_score")),
            "market_health_label": d.get("market_health_label", "Unknown"),
            "regime": d.get("regime", "Unknown"),
            "cape_ratio": _safe(d.get("cape_ratio")),
            "yield_curve": _safe(d.get("yield_spread_10y2y")),
            "vix": _safe(d.get("vix")),
            "pct_above_200sma": _safe(d.get("pct_above_200sma")),
            "sahm_rule": _safe(d.get("sahm_rule")),
            "consensus_tickers": consensus[:10],
        }

        out = data_dir / "market_pulse.json"
        out.write_text(json.dumps(pulse, indent=2), encoding="utf-8")
        log.info("  → market_pulse.json written")
    except Exception as e:
        log.error("market_pulse export failed: %s", e)
    finally:
        conn.close()


def _export_index(data_dir: Path, strategy: str, filename: str) -> None:
    """Export index holdings and performance for conservative or aggressive."""
    from db.schema import get_connection
    from indexes.performance import get_all_metrics
    conn = get_connection()
    today = date.today().isoformat()

    try:
        # Current holdings
        index_name = f"{strategy}_index"
        cur = conn.execute(
            """SELECT s.ticker, s.name, s.sector, ih.target_weight, ih.entry_date
               FROM index_holdings ih
               JOIN stocks s ON s.id = ih.stock_id
               WHERE ih.index_name = ? AND ih.exit_date IS NULL
               ORDER BY ih.target_weight DESC""",
            (index_name,)
        )
        holdings = [
            {
                "ticker": r["ticker"],
                "name": r["name"] or r["ticker"],
                "sector": r["sector"] or "Unknown",
                "weight": _safe(r["target_weight"]),
                "since": r["entry_date"],
            }
            for r in cur.fetchall()
        ]

        # Recent score for each holding
        scores_map = {}
        for h in holdings:
            cur2 = conn.execute(
                """SELECT s2.composite_score FROM scores s2
                   JOIN stocks st ON st.id = s2.stock_id
                   WHERE st.ticker = ? AND s2.strategy = ?
                   ORDER BY s2.score_date DESC LIMIT 1""",
                (h["ticker"], strategy)
            )
            row2 = cur2.fetchone()
            scores_map[h["ticker"]] = _safe(row2["composite_score"]) if row2 else None
        for h in holdings:
            h["score"] = scores_map.get(h["ticker"])

        # Performance metrics
        try:
            m = get_all_metrics(index_name)
        except Exception:
            m = {}

        # Last rebalance date
        cur3 = conn.execute(
            "SELECT MAX(entry_date) as last_rb FROM index_holdings WHERE index_name=?",
            (index_name,)
        )
        row3 = cur3.fetchone()
        last_rb = row3["last_rb"] if row3 else None

        out_data = {
            "as_of": today,
            "strategy": strategy,
            "holdings_count": len(holdings),
            "holdings": holdings,
            "performance": {
                "annual_return": _safe(m.get("annual_return")),
                "benchmark_return": _safe(m.get("benchmark_return")),
                "sharpe_ratio": _safe(m.get("sharpe_ratio")),
                "max_drawdown": _safe(m.get("max_drawdown")),
                "alpha": _safe(m.get("alpha")),
                "beta": _safe(m.get("beta")),
            },
            "last_rebalance": last_rb,
        }

        out = data_dir / filename
        out.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        log.info("  → %s written (%d holdings)", filename, len(holdings))
    except Exception as e:
        log.error("%s export failed: %s", filename, e)
    finally:
        conn.close()


def _export_conservative_index(data_dir: Path) -> None:
    _export_index(data_dir, "conservative", "conservative.json")


def _export_aggressive_index(data_dir: Path) -> None:
    _export_index(data_dir, "aggressive", "aggressive.json")


def _export_meta(data_dir: Path) -> None:
    """Export meta.json — site metadata and last-updated timestamp."""
    meta = {
        "site_name": "Horizon Ledger",
        "tagline": "Independent investment research — Not financial advice",
        "last_updated": date.today().isoformat(),
        "disclaimer": (
            "Horizon Ledger is an independent research tool for educational purposes. "
            "Nothing here is financial advice. Past performance does not guarantee future results. "
            "Always do your own research."
        ),
        "methodology_url": "about.html",
        "github_url": "https://github.com/Dividious/Horizon-Ledger-Public",
    }
    out = data_dir / "meta.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("  → meta.json written")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    export_all()
