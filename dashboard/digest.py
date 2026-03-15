"""
Horizon Ledger — Market Digest (Landing Page)
Daily macro environment summary: Market Health Score, CAPE, bubble flags,
sector heatmap, consensus top-25 picks, and auto-generated digest paragraph.

NOT FINANCIAL ADVICE.
"""

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    DISCLAIMER,
    CAPE_STRETCHED_THRESHOLD,
    CAPE_EXTREME_THRESHOLD,
    MARKET_HEALTH_WEIGHTS,
)
from db.schema import get_connection
from db.queries import get_latest_digest, get_digest_history, get_current_holdings


def show():
    st.title("🌍 Market Digest")
    st.caption(DISCLAIMER)

    conn = get_connection()
    digest = get_latest_digest(conn)
    conn.close()

    if digest is None:
        st.warning(
            "No market digest data yet. Run `python scripts/run_daily.py` to populate. "
            "After your first run, this page will show live macro data."
        )
        _show_empty_state()
        return

    digest_date = digest.get("date", date.today().isoformat())
    st.caption(f"Last updated: {digest_date}")

    # ── Row 1: Health Score + Regime ──────────────────────────────────────────
    col_traffic, col_components, col_regime = st.columns([1, 2, 1])

    health_score = digest.get("market_health_score") or 0
    health_label = digest.get("market_health_label") or "Unknown"
    regime       = digest.get("regime") or "unknown"

    health_emoji = {"Favorable": "🟢", "Caution": "🟡", "Risk-Off": "🔴"}.get(health_label, "⚪")
    regime_emoji = {"bull": "🟢", "neutral": "🟡", "bear": "🔴"}.get(regime, "⚪")

    with col_traffic:
        st.metric(
            label="Market Health",
            value=f"{health_emoji} {health_score}/100",
            delta=health_label,
        )
        st.metric(
            label="HMM Regime",
            value=f"{regime_emoji} {regime.title()}",
        )

    with col_components:
        st.markdown("**Component Breakdown**")
        component_data = _build_component_table(digest)
        if component_data:
            comp_df = pd.DataFrame(component_data)
            st.dataframe(comp_df, hide_index=True)

    with col_regime:
        # Gauge for health score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            title={"text": "Health Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#2ecc71" if health_score >= 65
                         else ("#f39c12" if health_score >= 40 else "#e74c3c")},
                "steps": [
                    {"range": [0,  40], "color": "#fdecea"},
                    {"range": [40, 65], "color": "#fef9e7"},
                    {"range": [65, 100], "color": "#eafaf1"},
                ],
                "threshold": {"line": {"color": "black", "width": 3},
                              "thickness": 0.8, "value": health_score},
            },
        ))
        fig_gauge.update_layout(height=200, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_gauge, width='stretch')

    st.divider()

    # ── Row 2: CAPE Gauge + Key Macro Numbers ──────────────────────────────────
    col_cape, col_macro = st.columns([1, 2])

    cape_ratio = digest.get("cape_ratio")
    cape_pct   = digest.get("cape_percentile")

    with col_cape:
        st.subheader("Shiller CAPE")
        if cape_ratio:
            # Get historical percentile markers
            conn2 = get_connection()
            from pipeline.macro import get_cape_stats
            cape_stats = get_cape_stats(conn2)
            conn2.close()

            p25 = cape_stats.get("historical_25th", 11.0)
            p50 = cape_stats.get("historical_50th", 17.0)
            p75 = cape_stats.get("historical_75th", 24.0)
            p90 = cape_stats.get("historical_90th", 32.0)
            exp_10y = cape_stats.get("expected_10y_return", 0.05)

            fig_cape = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cape_ratio,
                title={"text": f"CAPE ({cape_pct:.0f}th pct.)" if cape_pct else "CAPE"},
                gauge={
                    "axis": {"range": [5, 45]},
                    "bar":  {"color": "#e74c3c" if cape_ratio > CAPE_EXTREME_THRESHOLD
                             else ("#f39c12" if cape_ratio > CAPE_STRETCHED_THRESHOLD
                                   else "#2ecc71")},
                    "steps": [
                        {"range": [5,  p25], "color": "#eafaf1"},
                        {"range": [p25, p50], "color": "#fdfefe"},
                        {"range": [p50, p75], "color": "#fef9e7"},
                        {"range": [p75, 45],  "color": "#fdecea"},
                    ],
                    "threshold": {"line": {"color": "navy", "width": 3},
                                  "thickness": 0.8, "value": cape_ratio},
                },
            ))
            fig_cape.update_layout(height=220, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_cape, width='stretch')
            st.caption(
                f"25th/50th/75th/90th pct: {p25:.0f} / {p50:.0f} / {p75:.0f} / {p90:.0f}  "
                f"| Est. 10-year forward return: {exp_10y:.1%}"
            )
        else:
            st.info("CAPE data not yet loaded. Run `scripts/run_quarterly.py` to pull Shiller data.")

    with col_macro:
        st.subheader("Key Macro Indicators")
        slope  = digest.get("yield_curve_slope")
        spread = digest.get("credit_spread")
        vix    = digest.get("vix_level")
        tips   = digest.get("tips_breakeven")
        sahm   = digest.get("sahm_rule_value")
        pct200 = digest.get("pct_above_200sma")

        m1, m2, m3 = st.columns(3)
        m1.metric("Yield Curve (10Y-2Y)",
                  f"{slope*100:.0f} bps" if slope is not None else "N/A",
                  delta="Inverted ⚠️" if digest.get("yield_curve_inverted") else "Positive")
        m2.metric("HY Credit Spread",
                  f"{spread:.0f} bps" if spread is not None else "N/A")
        m3.metric("VIX",
                  f"{vix:.1f}" if vix is not None else "N/A",
                  delta="Elevated ⚠️" if (vix and vix > 25) else "Normal")

        m4, m5, m6 = st.columns(3)
        m4.metric("TIPS Breakeven (10Y)",
                  f"{tips:.2f}%" if tips is not None else "N/A")
        m5.metric("Sahm Rule",
                  f"{sahm:.2f}" if sahm is not None else "N/A",
                  delta="⚠️ Triggered" if digest.get("sahm_rule_triggered") else "Not triggered")
        m6.metric("% Stocks > 200-SMA",
                  f"{pct200:.0f}%" if pct200 is not None else "N/A",
                  delta=("Broad" if pct200 and pct200 > 70
                         else ("Narrow ⚠️" if pct200 and pct200 < 40 else "Moderate")))

    st.divider()

    # ── Row 3: Bubble Flags + Sparklines ──────────────────────────────────────
    col_flags, col_sparks = st.columns([1, 2])

    with col_flags:
        st.subheader("Active Risk Flags")
        bubble_flags = {}
        raw_flags = digest.get("bubble_flags")
        if raw_flags:
            try:
                bubble_flags = json.loads(raw_flags)
            except Exception:
                bubble_flags = {}

        if bubble_flags:
            for key, explanation in bubble_flags.items():
                key_label = key.replace("_", " ").title()
                st.warning(f"**{key_label}** — {explanation}", icon="⚠️")
        else:
            st.success("No major valuation or risk flags currently active.", icon="✅")

    with col_sparks:
        st.subheader("90-Day Macro Trends")
        _show_macro_sparklines()

    st.divider()

    # ── Row 4: Sector Valuation Heatmap ───────────────────────────────────────
    st.subheader("Sector Valuation Heatmap")
    st.caption(
        "Color coding vs. sector's own 5-year median valuations. "
        "🟢 Normal  🟡 1 metric stretched  🔴 2+ metrics stretched"
    )
    _show_sector_heatmap()

    st.divider()

    # ── Row 5: Consensus Top-25 ────────────────────────────────────────────────
    st.subheader("Cross-Strategy Consensus Top 25")
    st.caption(
        "Stocks ranked by combined percentile score across qualifying strategies. "
        "Requires scores from ≥ 2 strategies. 🟩 = currently in an index."
    )
    _show_consensus_table()

    st.divider()

    # ── Row 6: Auto-Generated Digest Paragraph ────────────────────────────────
    st.subheader("📋 Research Digest")
    digest_text = digest.get("digest_text")
    if digest_text:
        st.info(digest_text)
    else:
        st.caption("Digest text not yet generated.")

    # Mandatory disclaimer — always visible
    st.error(
        "⚠️ This digest is a heuristic summary for research purposes only. "
        "No signal shown here is a reliable buy or sell indicator. "
        "All macro indicators have historically produced false positives. "
        "NOT FINANCIAL ADVICE.",
        icon="⚠️",
    )

    # ── Historical health score trend ─────────────────────────────────────────
    with st.expander("Historical Health Score Trend", expanded=False):
        conn3 = get_connection()
        hist = get_digest_history(
            conn3,
            start=(date.today() - timedelta(days=365)).isoformat(),
        )
        conn3.close()
        if not hist.empty and "market_health_score" in hist.columns:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=hist["date"], y=hist["market_health_score"],
                mode="lines", name="Health Score", line=dict(color="#3498db", width=2),
            ))
            fig_hist.add_hline(y=65, line_dash="dot", line_color="green",
                                annotation_text="Favorable")
            fig_hist.add_hline(y=40, line_dash="dot", line_color="orange",
                                annotation_text="Caution")
            fig_hist.update_layout(
                height=280, yaxis=dict(range=[0, 100]),
                yaxis_title="Health Score", xaxis_title="Date",
            )
            st.plotly_chart(fig_hist, width='stretch')
        else:
            st.info("Not enough historical data yet (run daily for a few weeks).")


# ─── Helper renderers ─────────────────────────────────────────────────────────

def _build_component_table(digest: dict) -> list:
    """Build the sub-score breakdown table from raw digest data."""
    from pipeline.market_health import (
        _score_regime, _score_yield_curve, _score_credit_spread,
        _score_vix, _score_sahm_rule, _score_pct_above_200sma,
    )
    weights = MARKET_HEALTH_WEIGHTS
    sahm_triggered = bool(digest.get("sahm_rule_triggered"))
    slope  = digest.get("yield_curve_slope")
    spread = digest.get("credit_spread")
    vix    = digest.get("vix_level")
    sahm   = digest.get("sahm_rule_value")
    pct    = digest.get("pct_above_200sma")
    regime = digest.get("regime")

    rows = [
        {"Signal": "HMM Regime",        "Weight": f"{weights['regime']:.0%}",
         "Sub-Score": f"{_score_regime(regime):.0f}",
         "Contribution": f"{_score_regime(regime) * weights['regime']:.1f}"},
        {"Signal": "Yield Curve",       "Weight": f"{weights['yield_curve']:.0%}",
         "Sub-Score": f"{_score_yield_curve(slope):.0f}",
         "Contribution": f"{_score_yield_curve(slope) * weights['yield_curve']:.1f}"},
        {"Signal": "Credit Spreads",    "Weight": f"{weights['credit_spread']:.0%}",
         "Sub-Score": f"{_score_credit_spread(spread):.0f}",
         "Contribution": f"{_score_credit_spread(spread) * weights['credit_spread']:.1f}"},
        {"Signal": "VIX",               "Weight": f"{weights['vix']:.0%}",
         "Sub-Score": f"{_score_vix(vix):.0f}",
         "Contribution": f"{_score_vix(vix) * weights['vix']:.1f}"},
        {"Signal": "Sahm Rule",         "Weight": f"{weights['sahm_rule']:.0%}",
         "Sub-Score": f"{_score_sahm_rule(sahm_triggered):.0f}",
         "Contribution": f"{_score_sahm_rule(sahm_triggered) * weights['sahm_rule']:.1f}"},
        {"Signal": "% > 200-SMA",       "Weight": f"{weights['pct_above_200sma']:.0%}",
         "Sub-Score": f"{_score_pct_above_200sma(pct):.0f}",
         "Contribution": f"{_score_pct_above_200sma(pct) * weights['pct_above_200sma']:.1f}"},
    ]
    return rows


def _show_macro_sparklines():
    """Render small sparkline charts for VIX, yield curve, credit spread, TIPS."""
    try:
        conn = get_connection()
        from db.queries import get_macro_series
        start = (date.today() - timedelta(days=90)).isoformat()

        series_config = [
            ("VIXCLS",        "VIX", "red"),
            ("DGS10",         "10Y Treasury", "blue"),
            ("BAMLH0A0HYM2",  "HY Spread (bps)", "orange"),
            ("T10YIE",        "TIPS Breakeven", "purple"),
        ]

        cols = st.columns(len(series_config))
        for col, (series_id, label, color) in zip(cols, series_config):
            df = get_macro_series(conn, series_id, start=start)
            with col:
                if not df.empty:
                    current_val = df["value"].iloc[-1]
                    prior_val   = df["value"].iloc[0]
                    delta_val   = current_val - prior_val
                    st.metric(label, f"{current_val:.2f}",
                              delta=f"{delta_val:+.2f} (90d)")
                    fig = go.Figure(go.Scatter(
                        x=df["date"], y=df["value"],
                        mode="lines", line=dict(color=color, width=1.5),
                    ))
                    fig.update_layout(
                        height=80, margin=dict(t=0, b=0, l=0, r=0),
                        xaxis=dict(visible=False), yaxis=dict(visible=False),
                        showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.caption(f"{label}: no data")
        conn.close()
    except Exception as e:
        st.caption(f"Sparklines unavailable: {e}")


def _show_sector_heatmap():
    """Render sector valuation heatmap."""
    try:
        from pipeline.bubble_detector import get_sector_valuation_detail
        df = get_sector_valuation_detail()
        if df.empty:
            st.info("Sector valuation data not yet available. "
                    "Run `scripts/run_weekly.py` to populate.")
            return

        color_map = {"normal": "#eafaf1", "elevated": "#fef9e7", "stretched": "#fdecea"}
        status_emoji = {"normal": "🟢", "elevated": "🟡", "stretched": "🔴"}

        # Show as a styled table
        display_cols = ["sector", "status", "pe_current", "pe_hist",
                        "ps_current", "ps_hist", "ev_ebitda_current", "ev_ebitda_hist"]
        display = df[[c for c in display_cols if c in df.columns]].copy()
        display["status"] = display["status"].map(lambda s: f"{status_emoji.get(s, '⚪')} {s.title()}")

        for col in ["pe_current", "pe_hist", "ps_current", "ps_hist",
                    "ev_ebitda_current", "ev_ebitda_hist"]:
            if col in display.columns:
                display[col] = display[col].apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
                )

        display.columns = [c.replace("_current", " (Now)").replace("_hist", " (5Y Med)")
                            .replace("ev_ebitda", "EV/EBITDA").replace("pe", "P/E")
                            .replace("ps", "P/S").replace("sector", "Sector")
                            .replace("status", "Status")
                           for c in display.columns]

        st.dataframe(display, hide_index=True)

        # Compact heatmap chart
        if "sector" in df.columns and "stretched_count" in df.columns:
            df["color_val"] = df["stretched_count"]
            fig = px.bar(
                df.sort_values("stretched_count"),
                x="stretched_count", y="sector", orientation="h",
                color="stretched_count",
                color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
                range_color=[0, 3],
                labels={"stretched_count": "Stretched Metrics", "sector": "Sector"},
                title="Sector Stretch Count (0=normal, 3=all metrics stretched)",
            )
            fig.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, width='stretch')

    except Exception as e:
        st.caption(f"Sector heatmap unavailable: {e}")


def _show_consensus_table():
    """Render the cross-strategy consensus top-25 table."""
    try:
        from scoring.consensus import compute_consensus_top25
        conn = get_connection()

        # Get which tickers are in any index (for green highlight)
        in_any_index = set()
        for strategy in ["long_term", "dividend", "turnaround", "swing"]:
            h = get_current_holdings(conn, f"{strategy}_index")
            if not h.empty:
                in_any_index.update(h["ticker"].tolist())
        conn.close()

        with st.spinner("Computing consensus scores..."):
            df = compute_consensus_top25()

        if df.empty:
            st.info("No consensus data yet. Run scoring first (at least 2 strategies).")
            return

        # Add "in index" flag
        df["In Index"] = df["ticker"].apply(lambda t: "🟩" if t in in_any_index else "")

        display_cols = ["rank", "In Index", "ticker", "name", "sector",
                        "consensus_score", "n_strategies",
                        "long_term_score", "dividend_score",
                        "turnaround_score", "swing_score"]
        display = df[[c for c in display_cols if c in df.columns]].copy()
        for col in ["long_term_score", "dividend_score", "turnaround_score", "swing_score"]:
            if col in display.columns:
                display[col] = display[col].apply(
                    lambda x: f"{x:.0f}" if pd.notna(x) else "—"
                )
        display.columns = [c.replace("_score", "").replace("_", " ").title()
                           for c in display.columns]

        st.dataframe(display, hide_index=True)

        csv = df.drop(columns=["long_term_pct", "dividend_pct",
                                "turnaround_pct", "swing_pct"], errors="ignore").to_csv(index=False)
        st.download_button(
            label="📥 Export Consensus CSV",
            data=csv,
            file_name=f"consensus_top25_{date.today().isoformat()}.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.caption(f"Consensus table unavailable: {e}")


def _show_empty_state():
    """Show placeholder instructions when no data exists."""
    st.markdown("""
    ### Getting Started

    Run these commands to populate the Market Digest:

    ```cmd
    cd D:\\Stock_Analysis\\horizon-ledger
    .venv\\Scripts\\activate

    # One-time backfill (30-90 min)
    python scripts/backfill_history.py

    # Then run daily script to compute health score
    python scripts/run_daily.py
    ```

    After running, the digest will show:
    - 🟢/🟡/🔴 Market Health Score
    - CAPE ratio with historical percentile
    - Active risk flags
    - Sector valuation heatmap
    - Cross-strategy consensus top-25
    """)
