"""Horizon Ledger — Conservative & Aggressive Indexes Dashboard Page"""

import json
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from config import DISCLAIMER
from db.schema import get_connection
from db.queries import get_current_holdings, get_latest_scores


# ─── Next rebalance date helpers ─────────────────────────────────────────────

def _next_quarterly_rebalance() -> str:
    """Return the approximate next quarterly rebalance date (first trading day of Jan/Apr/Jul/Oct)."""
    today = date.today()
    month = today.month
    # Quarterly months: 1, 4, 7, 10
    quarterly_months = [1, 4, 7, 10]
    for qm in quarterly_months:
        candidate = date(today.year, qm, 1)
        if candidate > today:
            return candidate.isoformat()
    # Wrap to next year
    return date(today.year + 1, 1, 1).isoformat()


def _next_monthly_rebalance() -> str:
    """Return the approximate next monthly rebalance date (first trading day of next month)."""
    today = date.today()
    if today.month == 12:
        return date(today.year + 1, 1, 1).isoformat()
    return date(today.year, today.month + 1, 1).isoformat()


# ─── Per-index tab renderer ───────────────────────────────────────────────────

def _render_index_tab(
    conn,
    index_name: str,
    strategy: str,
    rebalance_label: str,
    next_rebalance: str,
) -> None:
    """Render a single index tab (holdings, metrics, chart, recent changes)."""

    # ── Fetch holdings ────────────────────────────────────────────────────────
    holdings_df = get_current_holdings(conn, index_name)

    # ── Fetch latest scores for the "Why It's Here" column ───────────────────
    scores_df = get_latest_scores(conn, strategy)

    if holdings_df.empty:
        st.warning(
            f"No holdings found for **{index_name}**. "
            "Run reconstitution (scripts/run_quarterly.py or run_weekly.py) first."
        )
        return

    # Build a score-components lookup: ticker → components dict
    score_lookup: dict = {}
    if not scores_df.empty:
        for _, srow in scores_df.iterrows():
            comp = {}
            if srow.get("score_components"):
                try:
                    comp = json.loads(srow["score_components"])
                except Exception:
                    pass
            score_lookup[srow["ticker"]] = {
                "composite_score": float(srow.get("composite_score") or 0),
                "components":      comp,
            }

    # ── Top-line metrics row ──────────────────────────────────────────────────
    n_holdings = len(holdings_df)

    # Paper portfolio metrics
    current_value = 0.0
    total_return_pct = 0.0
    vs_spy_pct = 0.0
    try:
        from paper_trading.engine import get_performance_metrics, get_portfolio_value
        portfolio_id = f"{strategy}_1000"
        current_value = get_portfolio_value(portfolio_id)
        metrics = get_performance_metrics(portfolio_id)
        combined = metrics.get("combined", {})
        total_return_pct = float(combined.get("total_return", 0.0)) * 100
        spy_ret = float(combined.get("benchmark_total_return", 0.0)) * 100
        vs_spy_pct = total_return_pct - spy_ret
    except Exception:
        pass

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Value (paper)", f"${current_value:,.2f}")
    m2.metric(
        "Total Return",
        f"{total_return_pct:+.2f}%",
        delta=f"{total_return_pct:+.2f}%",
        delta_color="normal",
    )
    m3.metric(
        "vs S&P 500",
        f"{vs_spy_pct:+.2f}%",
        delta=f"{vs_spy_pct:+.2f}%",
        delta_color="normal",
    )
    m4.metric("Holdings", n_holdings)

    st.divider()

    # ── Holdings table ────────────────────────────────────────────────────────
    st.subheader(f"{index_name.replace('_', ' ').title()} — Current Holdings")

    # Sort by target_weight descending
    disp = holdings_df.sort_values("target_weight", ascending=False).copy()

    # Add rank
    disp.insert(0, "Rank", range(1, len(disp) + 1))

    # Add Why It's Here
    from newsletter.sections import get_viability_explanation
    disp["Why It's Here"] = disp["ticker"].apply(
        lambda tkr: get_viability_explanation(
            score_lookup.get(tkr, {}).get("components", {})
        )
    )

    # Add composite score
    disp["Score"] = disp["ticker"].apply(
        lambda tkr: round(score_lookup.get(tkr, {}).get("composite_score", 0.0), 1)
    )

    # Format weight as percentage
    disp["Weight %"] = (disp["target_weight"] * 100).round(2).astype(str) + "%"

    display_cols = ["Rank", "ticker", "name", "Weight %", "Score", "Why It's Here"]
    col_rename = {"ticker": "Ticker", "name": "Company"}
    disp_final = disp[[c for c in display_cols if c in disp.columns]].rename(
        columns=col_rename
    )

    st.dataframe(disp_final, hide_index=True, use_container_width=True)

    # ── Weight distribution bar chart ─────────────────────────────────────────
    st.subheader("Weight Distribution")
    chart_df = disp.sort_values("target_weight", ascending=False).copy()
    chart_df["Weight (%)"] = (chart_df["target_weight"] * 100).round(2)
    fig = px.bar(
        chart_df,
        x="ticker",
        y="Weight (%)",
        color="sector" if "sector" in chart_df.columns else None,
        title=f"{index_name.replace('_', ' ').title()} — Target Weights",
        labels={"ticker": "Ticker", "Weight (%)": "Target Weight (%)"},
        height=360,
    )
    fig.update_layout(xaxis_tickangle=-45, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # ── Recent changes expander ───────────────────────────────────────────────
    with st.expander("Recent Rebalancing Changes (last 5)", expanded=False):
        try:
            changes_df = pd.read_sql(
                """SELECT rh.rebalance_date, rh.action, s.ticker, s.name,
                          rh.old_weight, rh.new_weight, rh.reason
                   FROM rebalancing_history rh
                   LEFT JOIN stocks s ON s.id = rh.stock_id
                   WHERE rh.index_name = ?
                   ORDER BY rh.rebalance_date DESC, rh.id DESC
                   LIMIT 5""",
                conn,
                params=[index_name],
            )
            if changes_df.empty:
                st.info("No rebalancing history recorded yet.")
            else:
                for col in ("old_weight", "new_weight"):
                    if col in changes_df.columns:
                        changes_df[col] = changes_df[col].apply(
                            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"
                        )
                st.dataframe(changes_df, hide_index=True, use_container_width=True)
        except Exception as e:
            st.caption(f"Rebalancing history unavailable: {e}")

    # ── Next rebalance info ───────────────────────────────────────────────────
    st.info(
        f"**Next {rebalance_label} Rebalance:** ~{next_rebalance}  "
        f"({'Quarterly' if 'quarterly' in rebalance_label.lower() else 'Monthly'} cadence)"
    )


# ─── Main page ────────────────────────────────────────────────────────────────

def show() -> None:
    st.title("📊 Conservative & Aggressive Indexes")
    st.caption(DISCLAIMER)

    conn = get_connection()

    tab_con, tab_agg = st.tabs(["🛡️ Conservative", "⚡ Aggressive"])

    with tab_con:
        _render_index_tab(
            conn=conn,
            index_name="conservative_index",
            strategy="conservative",
            rebalance_label="Quarterly",
            next_rebalance=_next_quarterly_rebalance(),
        )

    with tab_agg:
        _render_index_tab(
            conn=conn,
            index_name="aggressive_index",
            strategy="aggressive",
            rebalance_label="Monthly",
            next_rebalance=_next_monthly_rebalance(),
        )

    conn.close()
