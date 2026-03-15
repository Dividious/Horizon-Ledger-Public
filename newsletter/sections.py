"""
Horizon Ledger — Newsletter Section Generators
Each function returns an HTML string for one section of the newsletter.

NOT FINANCIAL ADVICE. All outputs are for informational/research purposes only.
"""

from typing import Optional

from config import DISCLAIMER


# ─── Factor Label Map ────────────────────────────────────────────────────────

FACTOR_LABELS = {
    "gross_profitability_pct":    "Quality",
    "roic_pct":                   "Capital Returns",
    "piotroski_f_pct":            "Balance Sheet",
    "altman_z_pct":               "Financial Safety",
    "altman_z_norm_pct":          "Financial Safety",
    "earnings_yield_pct":         "Value",
    "fcf_yield_pct":              "Cash Flow Value",
    "ev_to_ebitda_inv_pct":       "Value",
    "low_vol_252d_inv_pct":       "Low Volatility",
    "debt_to_equity_inv_pct":     "Low Leverage",
    "current_ratio_pct":          "Liquidity",
    "dividend_yield_pct":         "Dividend Yield",
    "div_growth_5y_pct":          "Dividend Growth",
    "consecutive_increases_pct":  "Dividend Consistency",
    "momentum_12m_pct":           "Momentum",
    "momentum_6m_pct":            "Momentum",
    "sector_momentum_pct":        "Sector Tailwind",
    "revenue_cagr_5y_pct":        "Revenue Growth",
    "earnings_cagr_5y_pct":       "Earnings Growth",
    "revenue_trajectory_pct":     "Revenue Acceleration",
    "rsi_signal_pct":             "Technical Setup",
    "volume_confirmation_pct":    "Volume Support",
    "price_to_book_inv_pct":      "Value",
}


# ─── Risk Flag Plain-English Explanations ────────────────────────────────────

FLAG_EXPLANATIONS = {
    "market_stretched": (
        "The Shiller CAPE ratio is above 30, suggesting below-average "
        "expected 10-year returns historically."
    ),
    "market_extreme": (
        "The Shiller CAPE ratio is above 35 — historically associated with "
        "very low or negative 10-year forward returns."
    ),
    "credit_complacency": (
        "High-yield credit spreads are near historical lows, suggesting "
        "investors may be underpricing default risk."
    ),
    "inversion_sustained": (
        "The yield curve has been inverted for over 90 days — the most "
        "reliable historical recession leading indicator."
    ),
}

# Known system-level flag keys (not sector names)
_SYSTEM_FLAG_KEYS = frozenset(FLAG_EXPLANATIONS.keys())


# ─── Internal Helpers ────────────────────────────────────────────────────────

def _safe_float(value, default: float = 0.0) -> float:
    """Return float(value) or default if value is None / non-numeric."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _pct_sign(value: float) -> str:
    """Format a float as a signed percentage string, e.g. +1.23% or -0.50%."""
    return f"{value:+.2f}%"


def _pct_sign1(value: float) -> str:
    """Format a float as a signed 1-decimal percentage string, e.g. +1.2%."""
    return f"{value:+.1f}%"


# ─── Public Helper ───────────────────────────────────────────────────────────

def get_viability_explanation(score_components: dict) -> str:
    """
    Map the top-2 factor names (by percentile value) to plain-English labels.

    Parameters
    ----------
    score_components : dict
        Keys are factor names ending in "_pct" (e.g. "gross_profitability_pct"),
        values are percentile scores (0–100).

    Returns
    -------
    str  e.g. "Quality + Value" or "Momentum + Revenue Growth"
             Returns "—" if score_components is empty or None.
    """
    if not score_components:
        return "—"

    # Sort factors by score descending
    sorted_factors = sorted(score_components.items(), key=lambda kv: _safe_float(kv[1]), reverse=True)

    seen_labels: list[str] = []
    for factor_name, _ in sorted_factors:
        label = FACTOR_LABELS.get(factor_name)
        if label and label not in seen_labels:
            seen_labels.append(label)
        if len(seen_labels) >= 2:
            break

    if not seen_labels:
        return "—"
    return " + ".join(seen_labels)


# ─── Section 1: Header ───────────────────────────────────────────────────────

def section_header(issue_date: str, issue_number: int) -> str:
    """
    Return HTML for the newsletter masthead.

    Parameters
    ----------
    issue_date   : str  "YYYY-MM-DD"
    issue_number : int
    """
    date_display = issue_date or "—"
    return f"""
<div class="newsletter-header">
  <div class="newsletter-logo">&#127749;</div>
  <h1 class="newsletter-title">HORIZON LEDGER WEEKLY</h1>
  <p class="newsletter-subtitle">Week of {date_display} &mdash; Issue #{issue_number}</p>
  <p class="newsletter-tagline">Personal Investment Research &bull; Not Financial Advice</p>
</div>
""".strip()


# ─── Section 2: Market Pulse ─────────────────────────────────────────────────

def section_market_pulse(market_health_data: dict) -> str:
    """
    Return HTML for the Market Pulse section.

    Parameters
    ----------
    market_health_data : dict  (row from market_digest_history)
        Keys used:
            market_health_score, market_health_label, digest_text,
            yield_curve_slope, yield_curve_inverted,
            vix_level, credit_spread, sahm_rule_value, sahm_rule_triggered,
            pct_above_200sma
    """
    if not market_health_data:
        return '<div class="section"><p class="no-data">Market pulse data unavailable.</p></div>'

    score = _safe_float(market_health_data.get("market_health_score"), 50)
    label = market_health_data.get("market_health_label") or "Unknown"
    digest = market_health_data.get("digest_text") or ""

    # Score emoji
    if score >= 65:
        emoji = "&#128994;"   # green circle
    elif score >= 40:
        emoji = "&#128993;"   # yellow circle
    else:
        emoji = "&#128308;"   # red circle

    # Signal values
    slope = market_health_data.get("yield_curve_slope")
    inverted = bool(market_health_data.get("yield_curve_inverted", 0))
    vix = market_health_data.get("vix_level")
    spread = market_health_data.get("credit_spread")
    sahm_val = market_health_data.get("sahm_rule_value")
    sahm_triggered = bool(market_health_data.get("sahm_rule_triggered", 0))
    pct_above = market_health_data.get("pct_above_200sma")

    # Yield curve row
    yc_status = "Inverted" if inverted else "Normal"
    yc_value = f"{slope:+.2f}%" if slope is not None else "N/A"

    # VIX row
    if vix is None:
        vix_status, vix_value = "Unknown", "N/A"
    elif vix >= 30:
        vix_status, vix_value = "High", f"{vix:.1f}"
    elif vix >= 20:
        vix_status, vix_value = "Elevated", f"{vix:.1f}"
    else:
        vix_status, vix_value = "Normal", f"{vix:.1f}"

    # Credit spread row
    if spread is None:
        cs_status, cs_value = "Unknown", "N/A"
    elif spread >= 500:
        cs_status, cs_value = "Wide", f"{spread:.0f} bps"
    elif spread >= 300:
        cs_status, cs_value = "Normal", f"{spread:.0f} bps"
    else:
        cs_status, cs_value = "Tight", f"{spread:.0f} bps"

    # Sahm rule row
    sahm_status = "Triggered" if sahm_triggered else "Clear"
    sahm_value_str = f"{sahm_val:.3f}" if sahm_val is not None else "N/A"

    # Breadth row
    if pct_above is None:
        breadth_status, breadth_value = "Unknown", "N/A"
    elif pct_above >= 60:
        breadth_status = "Broad"
        breadth_value = f"{pct_above:.0f}% above 200d SMA"
    else:
        breadth_status = "Narrow"
        breadth_value = f"{pct_above:.0f}% above 200d SMA"

    digest_html = f'<p class="digest-text">{digest}</p>' if digest else ""

    return f"""
<div class="section market-pulse-section">
  <div class="section-header">
    <span class="section-icon">&#128268;</span> MARKET PULSE
  </div>
  <div class="pulse-score-row">
    <span class="pulse-label">{label}</span>
    <span class="pulse-emoji">{emoji}</span>
    <span class="pulse-score">Score: {score:.0f}/100</span>
  </div>
  {digest_html}
  <table class="signal-table">
    <thead>
      <tr>
        <th>Signal</th>
        <th>Status</th>
        <th>Value</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Yield Curve</td>
        <td class="status-{"warn" if inverted else "ok"}">{yc_status}</td>
        <td>{yc_value}</td>
      </tr>
      <tr>
        <td>VIX</td>
        <td class="status-{"warn" if vix_status in ("High","Elevated") else "ok"}">{vix_status}</td>
        <td>{vix_value}</td>
      </tr>
      <tr>
        <td>Credit Spreads</td>
        <td class="status-{"warn" if cs_status == "Wide" else "ok"}">{cs_status}</td>
        <td>{cs_value}</td>
      </tr>
      <tr>
        <td>Sahm Rule</td>
        <td class="status-{"warn" if sahm_triggered else "ok"}">{sahm_status}</td>
        <td>{sahm_value_str}</td>
      </tr>
      <tr>
        <td>Market Breadth</td>
        <td class="status-{"warn" if breadth_status == "Narrow" else "ok"}">{breadth_status}</td>
        <td>{breadth_value}</td>
      </tr>
    </tbody>
  </table>
</div>
""".strip()


# ─── Section 3: Almanac ──────────────────────────────────────────────────────

def section_almanac(almanac_data: dict) -> str:
    """
    Return HTML for the Almanac section.

    Parameters
    ----------
    almanac_data : dict with keys:
        monthly_stats       : dict (from almanac.get_monthly_stats)
        sector_seasonality  : dict (from almanac.get_sector_seasonality)
        calendar            : list of dicts (from almanac.get_economic_calendar)
        cycle_position      : str
        cape_outlook        : dict {"median", "p25", "p75"}
        cape_ratio          : float
        cape_percentile     : float
        current_month       : int
        current_month_name  : str
    """
    if not almanac_data:
        return '<div class="section"><p class="no-data">Almanac data unavailable.</p></div>'

    monthly = almanac_data.get("monthly_stats") or {}
    seasonality = almanac_data.get("sector_seasonality") or {}
    calendar = almanac_data.get("calendar") or []
    cycle_pos = almanac_data.get("cycle_position") or "Market cycle data unavailable."
    cape_outlook = almanac_data.get("cape_outlook") or {}
    cape_ratio = almanac_data.get("cape_ratio")
    cape_pct = almanac_data.get("cape_percentile")
    month_name = almanac_data.get("current_month_name") or "This Month"

    # ── 1. Cycle position ─────────────────────────────────────────────────
    cycle_html = f'<p class="cycle-text">{cycle_pos}</p>'

    # ── 2. Monthly historical stats ───────────────────────────────────────
    pos_pct = _safe_int(monthly.get("positive_pct"), 0)
    avg_ret = _safe_float(monthly.get("avg_return"), 0.0)
    best = _safe_float(monthly.get("best"), 0.0)
    worst = _safe_float(monthly.get("worst"), 0.0)

    monthly_html = f"""
    <div class="almanac-monthly">
      <p>
        <strong>{month_name}</strong> has been positive in <strong>{pos_pct}%</strong>
        of years since 1950, with an average return of
        <strong>{avg_ret:+.1f}%</strong>.
        Best: <span class="positive">{best:+.1f}%</span> &bull;
        Worst: <span class="negative">{worst:+.1f}%</span>
      </p>
    </div>""".strip()

    # ── 3. Economic calendar (next ~30 days) ──────────────────────────────
    if calendar:
        cal_rows = ""
        for ev in calendar:
            ev_date = ev.get("date", "")
            ev_name = ev.get("event", "")
            ev_why = ev.get("why_it_matters", "")
            cal_rows += f"""
      <tr>
        <td class="cal-date">{ev_date}</td>
        <td class="cal-event">{ev_name}</td>
        <td class="cal-why">{ev_why}</td>
      </tr>"""
        calendar_html = f"""
    <table class="calendar-table">
      <thead>
        <tr><th>Date</th><th>Event</th><th>Why It Matters</th></tr>
      </thead>
      <tbody>{cal_rows}
      </tbody>
    </table>""".strip()
    else:
        calendar_html = '<p class="no-data">No major scheduled events in the coming month.</p>'

    # ── 4. Sector seasonality ─────────────────────────────────────────────
    favorable = seasonality.get("favorable") or []
    weak = seasonality.get("weak") or []

    fav_items = "".join(f'<li class="sector-favorable">{s}</li>' for s in favorable) or "<li>None</li>"
    weak_items = "".join(f'<li class="sector-weak">{s}</li>' for s in weak) or "<li>None</li>"

    seasonality_html = f"""
    <div class="seasonality-grid">
      <div class="seasonality-col">
        <h4 class="seasonality-header favorable-header">Seasonally Favorable</h4>
        <ul class="sector-list">{fav_items}</ul>
      </div>
      <div class="seasonality-col">
        <h4 class="seasonality-header weak-header">Seasonally Weak</h4>
        <ul class="sector-list">{weak_items}</ul>
      </div>
    </div>""".strip()

    # ── 5. 10-year CAPE outlook ───────────────────────────────────────────
    if cape_ratio is not None and cape_outlook:
        median = _safe_float(cape_outlook.get("median"), 0.0)
        p25 = _safe_float(cape_outlook.get("p25"), 0.0)
        p75 = _safe_float(cape_outlook.get("p75"), 0.0)
        pct_str = f"{cape_pct:.0f}th" if cape_pct is not None else "N/A"
        outlook_html = f"""
    <div class="cape-outlook">
      <p>
        From a current CAPE of <strong>{cape_ratio:.1f}</strong>
        ({pct_str} percentile historically):
        Median 10-year annualized return: <strong>{median:+.1f}%/yr</strong>
        &bull; Range: {p25:+.1f}% to {p75:+.1f}%
      </p>
      <p class="cape-disclaimer">
        This is a historically-derived heuristic, not a forecast. CAPE has
        limited short-term predictive power and wide confidence intervals.
      </p>
    </div>""".strip()
    else:
        outlook_html = '<p class="no-data">CAPE valuation data unavailable.</p>'

    return f"""
<div class="section almanac-section">
  <div class="section-header">
    <span class="section-icon">&#128197;</span> ALMANAC
  </div>

  <h3 class="almanac-subheader">Where We Are in the Cycle</h3>
  {cycle_html}

  <h3 class="almanac-subheader">This Month Historically ({month_name})</h3>
  {monthly_html}

  <h3 class="almanac-subheader">Coming Up (Next 30 Days)</h3>
  {calendar_html}

  <h3 class="almanac-subheader">Sector Seasonality</h3>
  {seasonality_html}

  <h3 class="almanac-subheader">10-Year Outlook <span class="not-forecast">(not a forecast)</span></h3>
  {outlook_html}
</div>
""".strip()


# ─── Internal: Index Section Builder ─────────────────────────────────────────

def _build_index_section(
    icon: str,
    title: str,
    index_data: dict,
    css_class: str,
) -> str:
    """Shared HTML builder for conservative and aggressive index sections."""
    if not index_data:
        return f'<div class="section {css_class}"><p class="no-data">{title} data unavailable.</p></div>'

    ret_week = _safe_float(index_data.get("return_week"), 0.0)
    ret_total = _safe_float(index_data.get("return_total"), 0.0)
    spy_total = _safe_float(index_data.get("spy_return_total"), 0.0)
    alpha = _safe_float(index_data.get("alpha"), 0.0)
    adds = _safe_int(index_data.get("adds"), 0)
    removes = _safe_int(index_data.get("removes"), 0)
    top5 = index_data.get("top5") or []

    # Holdings changes line
    if adds == 0 and removes == 0:
        changes_html = '<p class="holdings-changes no-changes">No changes this week.</p>'
    else:
        changes_html = (
            f'<p class="holdings-changes">'
            f'Holdings changes: <strong>{adds}</strong> added, '
            f'<strong>{removes}</strong> removed'
            f'</p>'
        )

    # Top 5 table
    if top5:
        rows = ""
        for holding in top5:
            rank = _safe_int(holding.get("rank"), 0)
            ticker = holding.get("ticker") or "—"
            name = holding.get("name") or "—"
            score = _safe_float(holding.get("score"), 0.0)
            components = holding.get("score_components") or {}
            explanation = get_viability_explanation(components)
            rows += f"""
      <tr>
        <td class="rank-cell">{rank}</td>
        <td class="ticker-cell">{ticker}</td>
        <td class="name-cell">{name}</td>
        <td class="score-cell">{score:.1f}</td>
        <td class="explain-cell">{explanation}</td>
      </tr>"""
        top5_html = f"""
    <table class="holdings-table">
      <thead>
        <tr>
          <th>Rank</th><th>Ticker</th><th>Company</th>
          <th>Score</th><th>Why It&rsquo;s Here</th>
        </tr>
      </thead>
      <tbody>{rows}
      </tbody>
    </table>""".strip()
    else:
        top5_html = '<p class="no-data">Top holdings data unavailable.</p>'

    vs_spy = ret_total - spy_total

    return f"""
<div class="section {css_class}">
  <div class="section-header">
    <span class="section-icon">{icon}</span> {title}
  </div>
  <div class="metric-row">
    <span class="metric-item">This week: <strong>{_pct_sign(ret_week)}</strong></span>
    <span class="metric-sep">&bull;</span>
    <span class="metric-item">Since inception: <strong>{_pct_sign1(ret_total)}</strong></span>
    <span class="metric-sep">&bull;</span>
    <span class="metric-item">vs S&amp;P 500: <strong>{_pct_sign1(vs_spy)}</strong></span>
  </div>
  {changes_html}
  <h4 class="top5-header">Top 5 Holdings</h4>
  {top5_html}
</div>
""".strip()


# ─── Section 4: Conservative Index ───────────────────────────────────────────

def section_conservative_index(index_data: dict) -> str:
    """
    Return HTML for the Conservative Index section.

    Parameters
    ----------
    index_data : dict
        return_week, return_total, spy_return_total, alpha, adds, removes,
        top5: list of {rank, ticker, name, score, score_components}
    """
    return _build_index_section(
        icon="&#128737;",          # shield
        title="CONSERVATIVE INDEX",
        index_data=index_data,
        css_class="conservative-section",
    )


# ─── Section 5: Aggressive Index ─────────────────────────────────────────────

def section_aggressive_index(index_data: dict) -> str:
    """
    Return HTML for the Aggressive Index section.

    Parameters
    ----------
    index_data : dict  (same structure as conservative)
    """
    return _build_index_section(
        icon="&#9889;",            # lightning bolt
        title="AGGRESSIVE INDEX",
        index_data=index_data,
        css_class="aggressive-section",
    )


# ─── Section 6: Top 10 Viability Scores ──────────────────────────────────────

def section_top10_viability(viability_data: dict) -> str:
    """
    Return HTML for the Top 10 Viability Scores section.

    Parameters
    ----------
    viability_data : dict
        "conservative" : list of {rank, ticker, name, score, score_components}
        "aggressive"   : list of {rank, ticker, name, score, score_components}
    """
    if not viability_data:
        return '<div class="section"><p class="no-data">Viability score data unavailable.</p></div>'

    def _build_table(stocks: list, label: str) -> str:
        if not stocks:
            return f'<p class="no-data">{label} viability data unavailable.</p>'
        rows = ""
        for stock in stocks[:5]:
            rank = _safe_int(stock.get("rank"), 0)
            ticker = stock.get("ticker") or "—"
            name = stock.get("name") or "—"
            score = _safe_float(stock.get("score"), 0.0)
            components = stock.get("score_components") or {}
            strengths = get_viability_explanation(components)
            rows += f"""
        <tr>
          <td class="rank-cell">{rank}</td>
          <td class="ticker-cell">{ticker}</td>
          <td class="name-cell">{name}</td>
          <td class="score-cell">{score:.1f}</td>
          <td class="strengths-cell">{strengths}</td>
        </tr>"""
        return f"""
      <table class="viability-table">
        <thead>
          <tr>
            <th>Rank</th><th>Ticker</th><th>Company</th>
            <th>Score</th><th>Key Strengths</th>
          </tr>
        </thead>
        <tbody>{rows}
        </tbody>
      </table>""".strip()

    conservative_stocks = viability_data.get("conservative") or []
    aggressive_stocks = viability_data.get("aggressive") or []

    con_table = _build_table(conservative_stocks, "Conservative")
    agg_table = _build_table(aggressive_stocks, "Aggressive")

    return f"""
<div class="section viability-section">
  <div class="section-header">
    <span class="section-icon">&#127942;</span> TOP 10 STOCK VIABILITY SCORES THIS WEEK
  </div>

  <h4 class="viability-subheader">&#128737; Conservative Strategy — Top 5</h4>
  {con_table}

  <h4 class="viability-subheader">&#9889; Aggressive Strategy — Top 5</h4>
  {agg_table}

  <p class="viability-note">
    Scores combine fundamental, technical, and macro factors.
    Higher score = stronger signal, not a buy recommendation.
  </p>
</div>
""".strip()


# ─── Section 7: Risk Flags ───────────────────────────────────────────────────

def section_risk_flags(bubble_flags: dict, market_health_data: dict) -> str:
    """
    Return HTML for the Risk Flags section.

    Parameters
    ----------
    bubble_flags       : dict  keys are flag identifiers (str), values typically True or a label
    market_health_data : dict  market_digest_history row (used for context, not rendered directly)
    """
    # Normalise inputs
    flags = bubble_flags or {}

    if not flags:
        return f"""
<div class="section risk-section">
  <div class="section-header">
    <span class="section-icon">&#9888;</span> RISK FLAGS
  </div>
  <div class="flag-box flag-box-ok">
    &#9989; No major valuation or risk flags currently active.
  </div>
</div>
""".strip()

    # Build flag list
    flag_items = ""
    for key, _value in flags.items():
        if key in FLAG_EXPLANATIONS:
            explanation = FLAG_EXPLANATIONS[key]
        else:
            # Treat as a sector flag
            sector_name = key.replace("_", " ").title()
            explanation = (
                f"The {sector_name} sector shows elevated valuations on multiple "
                f"metrics relative to its own 5-year history."
            )
        flag_items += f"""
    <li class="flag-item">
      <span class="flag-key">&#9888; {key.replace("_", " ").upper()}</span>
      <span class="flag-desc">{explanation}</span>
    </li>"""

    return f"""
<div class="section risk-section">
  <div class="section-header">
    <span class="section-icon">&#9888;</span> RISK FLAGS
  </div>
  <div class="flag-box flag-box-warning">
    <ul class="flag-list">{flag_items}
    </ul>
  </div>
</div>
""".strip()


# ─── Section 8: Watch Next Week ───────────────────────────────────────────────

def section_watch_next_week(calendar_data: list) -> str:
    """
    Return HTML for the "Watch Next Week" section.

    Parameters
    ----------
    calendar_data : list of {"date": str, "event": str, "why_it_matters": str}
    """
    if not calendar_data:
        return f"""
<div class="section watch-section">
  <div class="section-header">
    <span class="section-icon">&#128270;</span> WATCH NEXT WEEK
  </div>
  <p class="no-data">No major scheduled events next week.</p>
</div>
""".strip()

    items = ""
    for ev in calendar_data:
        ev_date = ev.get("date") or "—"
        ev_name = ev.get("event") or "—"
        ev_why = ev.get("why_it_matters") or ""
        items += f"""
    <li class="watch-item">
      <span class="watch-date">{ev_date}</span>
      <span class="watch-event"><strong>{ev_name}</strong></span>
      <span class="watch-why">{ev_why}</span>
    </li>"""

    return f"""
<div class="section watch-section">
  <div class="section-header">
    <span class="section-icon">&#128270;</span> WATCH NEXT WEEK
  </div>
  <ul class="watch-list">{items}
  </ul>
</div>
""".strip()


# ─── Section 9: Portfolio Update ─────────────────────────────────────────────

def section_portfolio_update(paper_portfolio_data: dict) -> str:
    """
    Return HTML for the Paper Portfolio Update section.

    Parameters
    ----------
    paper_portfolio_data : dict
        "conservative"  : {"week_return": float, "total_return": float, "spy_return": float}
        "aggressive"    : {"week_return": float, "total_return": float, "spy_return": float}
        "live_start_date" : str "YYYY-MM-DD"
    """
    if not paper_portfolio_data:
        return '<div class="section"><p class="no-data">Portfolio data unavailable.</p></div>'

    con = paper_portfolio_data.get("conservative") or {}
    agg = paper_portfolio_data.get("aggressive") or {}
    live_start = paper_portfolio_data.get("live_start_date") or "N/A"

    def _row(icon: str, label: str, pf: dict) -> str:
        week = _safe_float(pf.get("week_return"), 0.0)
        total = _safe_float(pf.get("total_return"), 0.0)
        spy = _safe_float(pf.get("spy_return"), 0.0)
        vs_spy = total - spy

        def _color_class(v: float) -> str:
            return "positive" if v >= 0 else "negative"

        return f"""
      <tr>
        <td>{icon} {label}</td>
        <td class="{_color_class(week)}">{_pct_sign(week)}</td>
        <td class="{_color_class(total)}">{_pct_sign1(total)}</td>
        <td class="{_color_class(vs_spy)}">{_pct_sign1(vs_spy)}</td>
      </tr>"""

    con_row = _row("&#128737;", "Conservative", con)
    agg_row = _row("&#9889;", "Aggressive", agg)

    return f"""
<div class="section portfolio-section">
  <div class="section-header">
    <span class="section-icon">&#128200;</span> PORTFOLIO UPDATE
  </div>
  <table class="portfolio-table">
    <thead>
      <tr>
        <th>Portfolio</th>
        <th>This Week</th>
        <th>Since Start</th>
        <th>vs S&amp;P 500</th>
      </tr>
    </thead>
    <tbody>{con_row}{agg_row}
    </tbody>
  </table>
  <p class="portfolio-note">
    Returns prior to {live_start} are backsimulated and subject to
    survivorship bias. Past performance does not guarantee future results.
  </p>
</div>
""".strip()


# ─── Section 10: Footer ───────────────────────────────────────────────────────

def section_footer() -> str:
    """
    Return HTML for the newsletter footer, including the mandatory disclaimer.
    """
    return f"""
<div class="newsletter-footer">
  <hr class="footer-divider" />
  <p class="disclaimer">{DISCLAIMER}</p>
  <p class="footer-credits">
    Horizon Ledger &bull; Personal Research System &bull; Built with Python,
    yfinance, SEC EDGAR, and FRED &bull; Free data only &bull;
    Scores generated by rule-based quantitative models.
  </p>
  <p class="footer-unsubscribe">
    This newsletter is generated locally and delivered to registered recipients only.
    Consult a licensed financial advisor before making any investment decisions.
  </p>
</div>
""".strip()
