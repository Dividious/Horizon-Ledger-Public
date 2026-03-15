"""
Horizon Ledger — Newsletter PDF Generator
Assembles all sections → HTML → WeasyPrint PDF

Usage:
    from newsletter.generator import generate_newsletter
    pdf_path = generate_newsletter("2026-03-22", 1)

NOT FINANCIAL ADVICE. All outputs are for informational/research purposes only.
"""

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from config import NEWSLETTER_DIR, LIVE_START_DATE, DISCLAIMER

log = logging.getLogger(__name__)

# ── WeasyPrint / fpdf2 availability ──────────────────────────────────────────

try:
    from weasyprint import HTML as _WeasyHTML
    _USE_WEASYPRINT = True
except ImportError:
    _USE_WEASYPRINT = False
    log.warning("WeasyPrint not available — falling back to fpdf2")


# ── PDF generation helpers ────────────────────────────────────────────────────

def _generate_pdf_fpdf2(html_content: str, output_path: Path) -> None:
    try:
        from fpdf import FPDF
        import re
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)
        text = re.sub(r'<[^>]+>', ' ', html_content)
        text = re.sub(r'\s+', ' ', text).strip()
        for line in text[:5000].split('. '):
            pdf.multi_cell(0, 5, line.strip())
        pdf.output(str(output_path))
    except Exception as e:
        log.error("fpdf2 fallback also failed: %s", e)
        # Write HTML file instead
        output_path.with_suffix('.html').write_text(html_content, encoding='utf-8')
        log.info("Saved newsletter as HTML instead: %s", output_path.with_suffix('.html'))


def _render_pdf(html_content: str, output_path: Path, css_path: Path) -> None:
    """Render HTML → PDF using WeasyPrint, falling back to fpdf2."""
    if _USE_WEASYPRINT:
        try:
            _WeasyHTML(string=html_content, base_url=str(css_path.parent)).write_pdf(
                str(output_path)
            )
            log.info("PDF written via WeasyPrint: %s", output_path)
            return
        except Exception as e:
            log.error("WeasyPrint failed: %s — trying fpdf2 fallback", e)

    _generate_pdf_fpdf2(html_content, output_path)


# ── Data loading helpers ──────────────────────────────────────────────────────

def _load_market_health(conn) -> dict:
    """Fetch the latest row from market_digest_history."""
    try:
        row = conn.execute(
            """SELECT * FROM market_digest_history
               ORDER BY date DESC LIMIT 1"""
        ).fetchone()
        if row:
            d = dict(row)
            # Parse bubble_flags JSON
            if d.get("bubble_flags"):
                try:
                    d["bubble_flags"] = json.loads(d["bubble_flags"])
                except Exception:
                    d["bubble_flags"] = {}
            else:
                d["bubble_flags"] = {}
            return d
    except Exception as e:
        log.warning("Could not load market_digest_history: %s", e)
    return {}


def _load_index_data(conn, index_name: str, strategy: str) -> dict:
    """
    Build the index_data dict expected by section_conservative_index /
    section_aggressive_index.
    """
    from db.queries import get_current_holdings, get_latest_scores

    try:
        holdings = get_current_holdings(conn, index_name)
        scores_df = get_latest_scores(conn, strategy)

        # Build top-5 list
        top5 = []
        if not scores_df.empty:
            scores_df = scores_df.reset_index(drop=True)
            scores_df["rank"] = range(1, len(scores_df) + 1)
            in_index = set(holdings["ticker"].tolist()) if not holdings.empty else set()
            in_index_scores = scores_df[scores_df["ticker"].isin(in_index)].head(5)
            for _, row in in_index_scores.iterrows():
                comp = {}
                if row.get("score_components"):
                    try:
                        comp = json.loads(row["score_components"])
                    except Exception:
                        comp = {}
                top5.append({
                    "rank":            int(row["rank"]),
                    "ticker":          row["ticker"],
                    "name":            row.get("name") or row["ticker"],
                    "score":           float(row["composite_score"] or 0),
                    "score_components": comp,
                })

        # Basic performance from paper portfolio
        week_ret = 0.0
        total_ret = 0.0
        spy_ret = 0.0
        alpha = 0.0
        adds = 0
        removes = 0

        try:
            from paper_trading.engine import get_performance_metrics
            portfolio_id = f"{strategy}_1000"
            metrics = get_performance_metrics(portfolio_id)
            combined = metrics.get("combined", {})
            week_ret  = float(combined.get("total_return", 0.0)) * 0.02  # approximate weekly slice
            total_ret = float(combined.get("total_return", 0.0)) * 100.0
            spy_ret   = float(combined.get("benchmark_total_return", 0.0)) * 100.0
            alpha     = total_ret - spy_ret
        except Exception:
            pass

        # Count recent adds/removes from rebalancing_history (last 7 days)
        try:
            week_ago = (date.today() - timedelta(days=7)).isoformat()
            adds_row = conn.execute(
                """SELECT COUNT(*) FROM rebalancing_history
                   WHERE index_name=? AND action='ADD' AND rebalance_date>=?""",
                (index_name, week_ago),
            ).fetchone()
            removes_row = conn.execute(
                """SELECT COUNT(*) FROM rebalancing_history
                   WHERE index_name=? AND action='REMOVE' AND rebalance_date>=?""",
                (index_name, week_ago),
            ).fetchone()
            adds    = adds_row[0] if adds_row else 0
            removes = removes_row[0] if removes_row else 0
        except Exception:
            pass

        return {
            "return_week":      week_ret,
            "return_total":     total_ret,
            "spy_return_total": spy_ret,
            "alpha":            alpha,
            "adds":             adds,
            "removes":          removes,
            "top5":             top5,
        }

    except Exception as e:
        log.warning("Could not load index data for %s: %s", index_name, e)
        return {}


def _load_viability_data(conn) -> dict:
    """Top 10 scores for conservative and aggressive strategies."""
    from db.queries import get_latest_scores

    def _top10(strategy: str) -> list:
        try:
            df = get_latest_scores(conn, strategy)
            if df.empty:
                return []
            df = df.reset_index(drop=True)
            df["rank"] = range(1, len(df) + 1)
            result = []
            for _, row in df.head(10).iterrows():
                comp = {}
                if row.get("score_components"):
                    try:
                        comp = json.loads(row["score_components"])
                    except Exception:
                        comp = {}
                result.append({
                    "rank":            int(row["rank"]),
                    "ticker":          row["ticker"],
                    "name":            row.get("name") or row["ticker"],
                    "score":           float(row["composite_score"] or 0),
                    "score_components": comp,
                })
            return result
        except Exception as e:
            log.warning("Could not load viability scores for %s: %s", strategy, e)
            return []

    return {
        "conservative": _top10("conservative"),
        "aggressive":   _top10("aggressive"),
    }


def _load_paper_portfolio_data() -> dict:
    """Load paper portfolio week/total returns for both public indexes."""
    result = {
        "conservative":   {},
        "aggressive":     {},
        "live_start_date": LIVE_START_DATE or date.today().isoformat(),
    }
    try:
        from paper_trading.engine import get_performance_metrics, get_equity_curve

        for strategy in ("conservative", "aggressive"):
            portfolio_id = f"{strategy}_1000"
            metrics = get_performance_metrics(portfolio_id)
            combined = metrics.get("combined", {})

            # Weekly return: last 5 business days from equity curve
            week_ret = 0.0
            try:
                week_ago = (date.today() - timedelta(days=7)).isoformat()
                curve = get_equity_curve(portfolio_id, start_date=week_ago)
                if not curve.empty and len(curve) >= 2:
                    v_start = curve.iloc[0]["portfolio_value"]
                    v_end   = curve.iloc[-1]["portfolio_value"]
                    if v_start and v_start > 0:
                        week_ret = (v_end - v_start) / v_start * 100
            except Exception:
                pass

            result[strategy] = {
                "week_return":   week_ret,
                "total_return":  float(combined.get("total_return", 0.0)) * 100,
                "spy_return":    float(combined.get("benchmark_total_return", 0.0)) * 100,
            }
    except Exception as e:
        log.warning("Could not load paper portfolio data: %s", e)

    return result


def _load_almanac_data(issue_date: str, market_health: dict) -> dict:
    """Assemble almanac section data."""
    from newsletter.almanac import (
        get_monthly_stats,
        get_sector_seasonality,
        get_economic_calendar,
        get_cycle_position,
        get_cape_10yr_outlook,
    )

    try:
        d = date.fromisoformat(issue_date)
    except Exception:
        d = date.today()

    month = d.month
    monthly_stats = get_monthly_stats(month)
    sector_seas   = get_sector_seasonality(month)
    calendar      = get_economic_calendar(issue_date, days=35)
    cycle_pos     = get_cycle_position(market_health) if market_health else ""
    cape_ratio    = market_health.get("cape_ratio") if market_health else None
    cape_pct      = market_health.get("cape_percentile") if market_health else None
    cape_outlook  = get_cape_10yr_outlook(float(cape_ratio)) if cape_ratio else {}

    return {
        "monthly_stats":      monthly_stats,
        "sector_seasonality": sector_seas,
        "calendar":           calendar,
        "cycle_position":     cycle_pos,
        "cape_outlook":       cape_outlook,
        "cape_ratio":         cape_ratio,
        "cape_percentile":    cape_pct,
        "current_month":      month,
        "current_month_name": monthly_stats.get("name", ""),
    }


def _load_next_week_calendar(issue_date: str) -> list:
    """Economic calendar for the 7 days after issue_date."""
    from newsletter.almanac import get_economic_calendar
    try:
        d = date.fromisoformat(issue_date)
        next_monday = (d + timedelta(days=1)).isoformat()
        return get_economic_calendar(next_monday, days=9)
    except Exception as e:
        log.warning("Could not load next-week calendar: %s", e)
        return []


# ── HTML assembly ─────────────────────────────────────────────────────────────

def _build_html(
    issue_date: str,
    issue_number: int,
    css_path: Path,
    market_health: dict,
    conservative_data: dict,
    aggressive_data: dict,
    viability_data: dict,
    paper_data: dict,
    almanac_data: dict,
    next_week_calendar: list,
) -> str:
    """Assemble the full newsletter HTML document."""
    from newsletter.sections import (
        section_header,
        section_market_pulse,
        section_almanac,
        section_conservative_index,
        section_aggressive_index,
        section_top10_viability,
        section_risk_flags,
        section_watch_next_week,
        section_portfolio_update,
        section_footer,
    )

    # Load CSS inline for WeasyPrint compatibility
    css_content = ""
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")

    bubble_flags = market_health.get("bubble_flags", {}) if market_health else {}

    sections_html = "\n\n".join([
        section_header(issue_date, issue_number),
        section_market_pulse(market_health),
        section_almanac(almanac_data),
        section_conservative_index(conservative_data),
        section_aggressive_index(aggressive_data),
        section_top10_viability(viability_data),
        section_risk_flags(bubble_flags, market_health),
        section_watch_next_week(next_week_calendar),
        section_portfolio_update(paper_data),
        section_footer(),
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Horizon Ledger Weekly — Issue #{issue_number} ({issue_date})</title>
  <style>
{css_content}
  </style>
</head>
<body>
{sections_html}
</body>
</html>"""


# ── DB record helper ──────────────────────────────────────────────────────────

def _record_in_db(conn, issue_date: str, issue_number: int, pdf_path: Path) -> None:
    """Insert or update a row in newsletter_history."""
    try:
        existing = conn.execute(
            "SELECT id FROM newsletter_history WHERE issue_date=?",
            (issue_date,),
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE newsletter_history
                   SET issue_number=?, pdf_path=?, status='generated'
                   WHERE issue_date=?""",
                (issue_number, str(pdf_path), issue_date),
            )
        else:
            conn.execute(
                """INSERT INTO newsletter_history
                   (issue_date, issue_number, pdf_path, status)
                   VALUES (?, ?, ?, 'generated')""",
                (issue_date, issue_number, str(pdf_path)),
            )
        conn.commit()
        log.info("Recorded newsletter in DB: issue %d (%s)", issue_number, issue_date)
    except Exception as e:
        log.warning("Could not record newsletter in DB: %s", e)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_newsletter(
    issue_date: str,
    issue_number: int,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate the weekly Horizon Ledger newsletter PDF.

    Parameters
    ----------
    issue_date   : str   "YYYY-MM-DD" — the Saturday date this issue covers
    issue_number : int   sequential issue number (e.g. 1, 2, 3 …)
    output_path  : Path  if None, auto-generate filename in NEWSLETTER_DIR

    Returns
    -------
    Path  — path to the generated PDF (or .html fallback)
    """
    NEWSLETTER_DIR.mkdir(parents=True, exist_ok=True)

    # Determine output path
    if output_path is None:
        try:
            d = date.fromisoformat(issue_date)
            week_num = d.isocalendar()[1]
            filename = f"horizon_ledger_{d.year}_{week_num:02d}.pdf"
        except Exception:
            filename = f"horizon_ledger_{issue_date.replace('-', '_')}.pdf"
        output_path = NEWSLETTER_DIR / filename

    css_path = Path(__file__).parent / "style.css"

    # ── Load all data ─────────────────────────────────────────────────────────
    from db.schema import get_connection
    conn = get_connection()

    log.info("Generating newsletter issue #%d for %s", issue_number, issue_date)

    market_health      = _load_market_health(conn)
    conservative_data  = _load_index_data(conn, "conservative_index", "conservative")
    aggressive_data    = _load_index_data(conn, "aggressive_index", "aggressive")
    viability_data     = _load_viability_data(conn)
    almanac_data       = _load_almanac_data(issue_date, market_health)
    next_week_calendar = _load_next_week_calendar(issue_date)

    conn.close()

    paper_data = _load_paper_portfolio_data()

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    html = _build_html(
        issue_date=issue_date,
        issue_number=issue_number,
        css_path=css_path,
        market_health=market_health,
        conservative_data=conservative_data,
        aggressive_data=aggressive_data,
        viability_data=viability_data,
        paper_data=paper_data,
        almanac_data=almanac_data,
        next_week_calendar=next_week_calendar,
    )

    # ── Render PDF ────────────────────────────────────────────────────────────
    _render_pdf(html, output_path, css_path)

    # ── Record in DB ──────────────────────────────────────────────────────────
    from db.schema import get_connection as _gc
    conn2 = _gc()
    _record_in_db(conn2, issue_date, issue_number, output_path)
    conn2.close()

    # Return actual file written (may be .html if both PDF renderers failed)
    if not output_path.exists():
        html_fallback = output_path.with_suffix(".html")
        if html_fallback.exists():
            log.warning("PDF not created — returning HTML fallback: %s", html_fallback)
            return html_fallback

    log.info("Newsletter generation complete: %s", output_path)
    return output_path


def get_next_issue_number(conn) -> int:
    """
    Return the next sequential issue number based on newsletter_history.
    Returns 1 if no prior issues exist.
    """
    try:
        row = conn.execute(
            "SELECT MAX(issue_number) FROM newsletter_history"
        ).fetchone()
        max_num = row[0] if row and row[0] is not None else 0
        return max_num + 1
    except Exception:
        return 1


def get_most_recent_saturday(reference: Optional[date] = None) -> str:
    """
    Return the most recent Saturday on or before reference date as "YYYY-MM-DD".
    Defaults to today.
    """
    if reference is None:
        reference = date.today()
    days_since_saturday = (reference.weekday() - 5) % 7
    last_saturday = reference - timedelta(days=days_since_saturday)
    return last_saturday.isoformat()
