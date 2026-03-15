"""
Horizon Ledger — Paper Portfolio Simulator (Page 7)
Simulates $1,000 invested in each strategy index from inception.

Key separation:
  - Backsimulated period (before LIVE_START_DATE): survivorship bias applies,
    shown with orange shading and prominent warning.
  - Live period (from LIVE_START_DATE): genuine out-of-sample tracking.

SIMULATION ONLY. Not financial advice.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DISCLAIMER, LIVE_START_DATE, PAPER_STARTING_CASH, PAPER_SLIPPAGE
from db.schema import get_connection
from db.queries import get_all_paper_portfolios, get_open_paper_positions, get_price_on_date


STRATEGY_LABELS = {
    "long_term":  "📈 Long-Term Buy & Hold",
    "dividend":   "💰 Dividend / Income",
    "turnaround": "🔄 High-Risk Turnaround",
    "swing":      "⚡ Short-Term Swing",
}


def show():
    st.title("📈 Paper Portfolio Simulator")
    st.caption(DISCLAIMER)

    # Mandatory disclaimer banner — always visible
    live_str = LIVE_START_DATE or date.today().isoformat()
    st.error(
        f"**Backsimulated returns** reflect historical prices applied to the **current** stock "
        f"universe and are subject to **survivorship bias**. Only returns from "
        f"**{live_str} onward** represent genuine out-of-sample performance. "
        "This is a research simulation, not financial advice.",
        icon="⚠️",
    )

    conn = get_connection()
    portfolios = get_all_paper_portfolios(conn)
    conn.close()

    if portfolios.empty:
        _show_setup_instructions()
        return

    # ── Portfolio selector ─────────────────────────────────────────────────────
    portfolio_ids = portfolios["portfolio_id"].tolist()
    selected_id = st.selectbox(
        "Select portfolio:",
        options=["All (comparison view)"] + portfolio_ids,
        format_func=lambda x: x if x == "All (comparison view)"
                               else _portfolio_label(portfolios, x),
    )

    st.divider()

    if selected_id == "All (comparison view)":
        _show_comparison_view(portfolios)
    else:
        _show_single_portfolio(selected_id, portfolios)


def _portfolio_label(portfolios: pd.DataFrame, portfolio_id: str) -> str:
    row = portfolios[portfolios["portfolio_id"] == portfolio_id]
    if row.empty:
        return portfolio_id
    strategy = row.iloc[0]["strategy"]
    return STRATEGY_LABELS.get(strategy, strategy)


def _show_comparison_view(portfolios: pd.DataFrame):
    """Show all four portfolios on a single equity curve chart."""
    st.subheader("All Portfolios — $1,000 Starting Value Comparison")

    from paper_trading.engine import get_equity_curve

    fig = go.Figure()
    color_map = {
        "long_term":  "#3498db",
        "dividend":   "#2ecc71",
        "turnaround": "#e74c3c",
        "swing":      "#f39c12",
    }

    live_start_date = LIVE_START_DATE or date.today().isoformat()
    spy_added = False

    for _, prow in portfolios.iterrows():
        pid      = prow["portfolio_id"]
        strategy = prow["strategy"]
        label    = STRATEGY_LABELS.get(strategy, strategy)
        color    = color_map.get(strategy, "#888")

        curve = get_equity_curve(pid)
        if curve.empty or "portfolio_value_norm" not in curve.columns:
            continue

        fig.add_trace(go.Scatter(
            x=curve["date"],
            y=curve["portfolio_value_norm"],
            mode="lines", name=label,
            line=dict(color=color, width=2),
        ))

        # Add SPY once
        if not spy_added and "spy_value" in curve.columns:
            spy = curve[curve["spy_value"].notna()]
            if not spy.empty:
                fig.add_trace(go.Scatter(
                    x=spy["date"], y=spy["spy_value"],
                    mode="lines", name="SPY Benchmark",
                    line=dict(color="gray", width=1.5, dash="dot"),
                ))
                spy_added = True

    # Shaded backsim region — convert strings to pd.Timestamp so Plotly's
    # annotation code doesn't try to sum mixed int/str x-values.
    import pandas as _pd
    live_start_ts  = _pd.Timestamp(live_start_date)
    created_min    = portfolios["created_date"].min() or "2021-01-01"
    created_min_ts = _pd.Timestamp(created_min)

    # Only draw backsim region if there's a meaningful gap before live start
    has_backsim_period = created_min_ts < live_start_ts
    if has_backsim_period:
        fig.add_vrect(
            x0=created_min_ts,
            x1=live_start_ts,
            fillcolor="rgba(255,165,0,0.08)",
            layer="below", line_width=0,
        )
        fig.add_annotation(
            x=created_min_ts, y=0.97, yref="paper",
            text="◀ Backsimulated", showarrow=False,
            xanchor="left", font=dict(color="orange", size=10),
        )
    fig.add_shape(
        type="line", x0=live_start_ts, x1=live_start_ts,
        y0=0, y1=1, yref="paper",
        line=dict(color="orange", dash="dash", width=1),
    )
    fig.add_annotation(
        x=live_start_ts, y=0.97, yref="paper",
        text="Live tracking begins ▶", showarrow=False,
        xanchor="left", font=dict(color="orange", size=10),
    )

    fig.update_layout(
        height=450, yaxis_title="Portfolio Value ($)", xaxis_title="Date",
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, width='stretch')

    # Summary metrics table
    st.subheader("Summary Comparison")
    _show_summary_table(portfolios)


def _show_single_portfolio(portfolio_id: str, portfolios: pd.DataFrame):
    """Show detailed view for a single portfolio."""
    from paper_trading.engine import get_equity_curve, get_performance_metrics

    prow = portfolios[portfolios["portfolio_id"] == portfolio_id].iloc[0]
    strategy    = prow["strategy"]
    live_start  = prow["live_start_date"] or date.today().isoformat()
    start_cash  = float(prow["starting_cash"] or PAPER_STARTING_CASH)

    label = STRATEGY_LABELS.get(strategy, strategy)
    st.subheader(f"{label} — Portfolio Detail")

    # ── Snapshot metrics ─────────────────────────────────────────────────────
    from paper_trading.engine import get_portfolio_value
    current_value = get_portfolio_value(portfolio_id)
    total_return  = (current_value / start_cash) - 1 if start_cash > 0 else 0

    # SPY return for same period
    conn = get_connection()
    spy_id = conn.execute("SELECT id FROM stocks WHERE ticker='SPY'").fetchone()
    spy_return_str = "N/A"
    if spy_id:
        spy_start_p = get_price_on_date(conn, spy_id["id"], prow["created_date"] or live_start)
        spy_now_p   = get_price_on_date(conn, spy_id["id"], date.today().isoformat())
        if spy_start_p and spy_now_p and spy_start_p > 0:
            spy_ret = (spy_now_p - spy_start_p) / spy_start_p
            spy_return_str = f"{spy_ret:+.2%}"
    conn.close()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Starting Value", f"${start_cash:,.2f}")
    col2.metric("Current Value",  f"${current_value:,.2f}",
                delta=f"{total_return:+.2%}")
    col3.metric("vs SPY",         spy_return_str)
    col4.metric("Slippage Rate",  f"{PAPER_SLIPPAGE:.3%} per trade")

    # ── Equity curve ─────────────────────────────────────────────────────────
    st.subheader("Equity Curve vs SPY")
    curve = get_equity_curve(portfolio_id)

    if curve.empty or "portfolio_value_norm" not in curve.columns:
        st.info("No equity curve data yet. Sync portfolio to index first.")
    else:
        fig = go.Figure()

        # Backsim shading
        backsim_mask = curve["is_backsimulated"] == 1
        if backsim_mask.any():
            bs_end = curve[backsim_mask]["date"].max()
            fig.add_vrect(
                x0=curve["date"].min(), x1=bs_end,
                fillcolor="rgba(255,165,0,0.10)",
                layer="below", line_width=0,
            )
            fig.add_annotation(
                x=curve[backsim_mask]["date"].median(),
                y=curve["portfolio_value_norm"].max() * 0.95,
                text="⚠️ Backsimulated<br>(survivorship bias)",
                showarrow=False, font=dict(size=10, color="orange"),
            )

        # Portfolio line
        fig.add_trace(go.Scatter(
            x=curve["date"], y=curve["portfolio_value_norm"],
            mode="lines", name=label,
            line=dict(color="#3498db", width=2),
        ))

        # Vertical line at live start
        import pandas as _pd
        live_start_ts = _pd.Timestamp(live_start)
        fig.add_shape(
            type="line", x0=live_start_ts, x1=live_start_ts,
            y0=0, y1=1, yref="paper",
            line=dict(color="orange", dash="dash", width=1),
        )
        fig.add_annotation(
            x=live_start_ts, y=1, yref="paper",
            text="Live tracking begins", showarrow=False,
            xanchor="left", font=dict(color="orange", size=11),
        )

        # SPY line
        if "spy_value" in curve.columns:
            spy = curve[curve["spy_value"].notna()]
            if not spy.empty:
                fig.add_trace(go.Scatter(
                    x=spy["date"], y=spy["spy_value"],
                    mode="lines", name="SPY",
                    line=dict(color="gray", width=1.5, dash="dot"),
                ))

        fig.update_layout(
            height=400, yaxis_title="Portfolio Value ($)", xaxis_title="Date",
            hovermode="x unified",
        )
        st.plotly_chart(fig, width='stretch')

    # ── Current Holdings ──────────────────────────────────────────────────────
    st.subheader("Current Holdings")
    conn = get_connection()
    positions = get_open_paper_positions(conn, portfolio_id)
    today_str = date.today().isoformat()

    if positions.empty:
        st.info("No open positions. Use the Rebalancing tab to sync the portfolio to the index.")
    else:
        display_rows = []
        for _, pos in positions.iterrows():
            current_p = get_price_on_date(conn, int(pos["stock_id"]), today_str)
            shares    = float(pos["shares"] or 0)
            cost      = float(pos["cost_basis"] or 0)
            curr_val  = (current_p or 0) * shares
            gain      = curr_val - (cost * shares)
            gain_pct  = gain / (cost * shares) if (cost and cost > 0 and shares > 0) else None

            display_rows.append({
                "Ticker":       pos["ticker"],
                "Company":      pos.get("name", ""),
                "Shares":       f"{shares:.4f}",
                "Cost Basis":   f"${cost:.2f}",
                "Current Price": f"${current_p:.2f}" if current_p else "N/A",
                "Current Value": f"${curr_val:,.2f}" if current_p else "N/A",
                "Gain/Loss":    f"${gain:+,.2f}" if current_p else "N/A",
                "Gain/Loss %":  f"{gain_pct:+.2%}" if gain_pct is not None else "N/A",
            })
        conn.close()
        st.dataframe(pd.DataFrame(display_rows), hide_index=True)

    if not positions.empty:
        conn.close() if not conn else None

    # ── Transaction Log ───────────────────────────────────────────────────────
    st.subheader("Transaction Log")

    from db.queries import get_paper_transactions
    conn2 = get_connection()
    tx_df = get_paper_transactions(conn2, portfolio_id)
    conn2.close()

    if tx_df.empty:
        st.info("No transactions recorded yet.")
    else:
        # Filters
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            tx_types = ["All"] + sorted(tx_df["type"].unique().tolist())
            sel_type = st.selectbox("Transaction type:", tx_types)
        with fcol2:
            backsim_opt = st.selectbox("Period:", ["All", "Backsimulated only", "Live only"])
        with fcol3:
            date_range = st.date_input(
                "Date range:",
                value=(date.today() - timedelta(days=365), date.today()),
            )

        filtered_tx = tx_df.copy()
        if sel_type != "All":
            filtered_tx = filtered_tx[filtered_tx["type"] == sel_type]
        if backsim_opt == "Backsimulated only":
            filtered_tx = filtered_tx[filtered_tx["is_backsimulated"] == 1]
        elif backsim_opt == "Live only":
            filtered_tx = filtered_tx[filtered_tx["is_backsimulated"] == 0]
        if len(date_range) == 2:
            start_d, end_d = [d.isoformat() for d in date_range]
            filtered_tx = filtered_tx[
                (filtered_tx["date"] >= start_d) & (filtered_tx["date"] <= end_d)
            ]

        display_tx = filtered_tx[[
            "date", "type", "ticker", "shares", "price", "execution_price",
            "cash_impact", "reason", "is_backsimulated"
        ]].copy() if not filtered_tx.empty else pd.DataFrame()

        if not display_tx.empty:
            display_tx["shares"]          = display_tx["shares"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            display_tx["price"]           = display_tx["price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
            display_tx["execution_price"] = display_tx["execution_price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
            display_tx["cash_impact"]     = display_tx["cash_impact"].apply(lambda x: f"${x:+,.2f}" if pd.notna(x) else "N/A")
            display_tx["is_backsimulated"] = display_tx["is_backsimulated"].map({1: "⚠️ Backsim", 0: "✅ Live"})

        st.dataframe(display_tx if not display_tx.empty else pd.DataFrame(), hide_index=True)

        if not filtered_tx.empty:
            csv = filtered_tx.to_csv(index=False)
            st.download_button("📥 Export Transactions CSV", csv,
                               f"transactions_{portfolio_id}_{date.today().isoformat()}.csv",
                               "text/csv")

    # ── Performance Metrics ───────────────────────────────────────────────────
    st.subheader("Performance Metrics")
    metrics = get_performance_metrics(portfolio_id)

    if not metrics:
        st.info("Insufficient data for performance metrics (need at least 5 trading days).")
    else:
        periods = [k for k in ["combined", "live", "backsimulated"] if k in metrics]
        if periods:
            rows = [
                "total_return", "annualized_return", "volatility",
                "sharpe", "sortino", "max_drawdown", "beta", "alpha",
            ]
            metric_labels = {
                "total_return": "Total Return", "annualized_return": "Ann. Return",
                "volatility": "Volatility", "sharpe": "Sharpe Ratio",
                "sortino": "Sortino Ratio", "max_drawdown": "Max Drawdown",
                "beta": "Beta (vs SPY)", "alpha": "Alpha (ann.)",
            }
            pct_metrics  = {"total_return", "annualized_return", "volatility", "max_drawdown", "alpha"}
            metric_table = {"Metric": [metric_labels.get(r, r) for r in rows]}
            for period in periods:
                p_data = metrics.get(period, {})
                def _fmt(r, v):
                    if v is None:
                        return "N/A"
                    if r in pct_metrics:
                        return f"{v:.2%}"
                    return f"{v:.3f}"
                metric_table[period.title()] = [_fmt(r, p_data.get(r)) for r in rows]
                if period == "backsimulated":
                    metric_table[f"{period.title()} ⚠️"] = metric_table.pop(period.title())

            st.dataframe(pd.DataFrame(metric_table), hide_index=True)

    # Sync button
    st.divider()
    st.subheader("Sync to Index")
    st.caption("Mirror the current index holdings to this portfolio.")
    if st.button("🔄 Sync Portfolio to Index Now", key=f"sync_{portfolio_id}"):
        with st.spinner("Syncing..."):
            try:
                from paper_trading.engine import sync_portfolio_to_index
                result = sync_portfolio_to_index(portfolio_id, strategy)
                st.success(f"Synced! {len(result.get('buys', []))} buys, "
                           f"{len(result.get('sells', []))} sells.")
                st.rerun()
            except Exception as e:
                st.error(f"Sync failed: {e}")


def _show_summary_table(portfolios: pd.DataFrame):
    """Show a concise performance summary across all portfolios."""
    from paper_trading.engine import get_portfolio_value, get_performance_metrics

    rows = []
    for _, prow in portfolios.iterrows():
        pid       = prow["portfolio_id"]
        strategy  = prow["strategy"]
        start_c   = float(prow["starting_cash"] or PAPER_STARTING_CASH)
        curr_val  = get_portfolio_value(pid)
        total_ret = (curr_val / start_c) - 1 if start_c > 0 else 0

        metrics   = get_performance_metrics(pid)
        combined  = metrics.get("combined", {})
        live      = metrics.get("live", {})

        rows.append({
            "Strategy":       STRATEGY_LABELS.get(strategy, strategy),
            "Start Value":    f"${start_c:,.0f}",
            "Current Value":  f"${curr_val:,.2f}",
            "Total Return":   f"{total_ret:+.2%}",
            "Ann. Return":    f"{combined.get('annualized_return', 0):+.2%}" if combined.get("annualized_return") else "N/A",
            "Sharpe":         f"{combined.get('sharpe', 0):.2f}" if combined.get("sharpe") else "N/A",
            "Max Drawdown":   f"{combined.get('max_drawdown', 0):.2%}" if combined.get("max_drawdown") else "N/A",
            "Live Return":    f"{live.get('total_return', 0):+.2%}" if live.get("total_return") else "N/A",
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True)


def _show_setup_instructions():
    """Show setup instructions when no portfolios exist."""
    st.info(
        "No paper portfolios found. Initialize them by running the quarterly script or "
        "calling `paper_trading.engine.initialize_all_portfolios()` once."
    )
    st.markdown("""
    ```cmd
    cd D:\\Stock_Analysis\\horizon-ledger
    .venv\\Scripts\\activate
    python -c "from paper_trading.engine import initialize_all_portfolios; initialize_all_portfolios()"
    ```

    Then run `scripts/run_quarterly.py` to sync the portfolios to the current index holdings.

    **Before you start**, set `LIVE_START_DATE` in `config.py` to today's date:
    ```python
    LIVE_START_DATE = "2026-03-15"  # Your actual live start date
    ```
    Everything before this date will be labeled as backsimulated.
    """)
