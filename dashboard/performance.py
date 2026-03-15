"""
Horizon Ledger — Performance Tracking Dashboard Page
Cumulative returns, Sharpe, drawdown, alpha decay, factor attribution.
"""

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from config import BENCHMARK_TICKER, DISCLAIMER


def show():
    st.title("📉 Performance Tracking")
    st.caption(DISCLAIMER)

    strategies = ["long_term", "dividend", "turnaround", "swing"]
    strategy_labels = {
        "long_term":  "Long-Term Buy & Hold",
        "dividend":   "Dividend / Income",
        "turnaround": "High-Risk Turnaround",
        "swing":      "Short-Term Swing",
    }

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Date Range")
        today = date.today()
        start_date = st.date_input("From:", value=today - timedelta(days=365))
        end_date   = st.date_input("To:",   value=today)
        selected_strategies = st.multiselect(
            "Strategies:",
            options=strategies,
            default=strategies,
            format_func=lambda s: strategy_labels[s],
        )

    start_str = start_date.isoformat()
    end_str   = end_date.isoformat()

    # ── Cumulative Return Chart ────────────────────────────────────────────────
    st.subheader("Cumulative Return vs. SPY")
    try:
        from indexes.performance import compute_index_returns, get_benchmark_returns

        fig = go.Figure()
        for strategy in selected_strategies:
            index_name = f"{strategy}_index"
            try:
                idx_series = compute_index_returns(index_name, start_str, end_str)
                if not idx_series.empty:
                    fig.add_trace(go.Scatter(
                        x=idx_series.index,
                        y=(idx_series - 1) * 100,
                        name=strategy_labels[strategy],
                        mode="lines",
                    ))
            except Exception:
                pass

        # Benchmark
        bmk = get_benchmark_returns(start_str, end_str)
        if not bmk.empty:
            fig.add_trace(go.Scatter(
                x=bmk.index,
                y=(bmk - 1) * 100,
                name=f"{BENCHMARK_TICKER} (Benchmark)",
                line=dict(dash="dash", color="gray"),
                mode="lines",
            ))

        fig.update_layout(
            yaxis_title="Return (%)",
            xaxis_title="Date",
            height=450,
            hovermode="x unified",
        )
        st.plotly_chart(fig, width='stretch')
    except Exception as e:
        st.error(f"Could not compute returns: {e}")

    # ── Metrics Table ─────────────────────────────────────────────────────────
    st.subheader("Performance Metrics")
    try:
        from indexes.performance import get_all_metrics
        metrics_rows = []
        for strategy in selected_strategies:
            index_name = f"{strategy}_index"
            try:
                m = get_all_metrics(index_name, start_str, end_str)
                metrics_rows.append({
                    "Strategy":        strategy_labels[strategy],
                    "Ann. Return":     f"{m.get('annual_return', 0):+.1%}",
                    "Volatility":      f"{m.get('annual_volatility', 0):.1%}",
                    "Sharpe":          f"{m.get('sharpe_ratio', 0):.2f}",
                    "Sortino":         f"{m.get('sortino_ratio', 0):.2f}",
                    "Max DD":          f"{m.get('max_drawdown', 0):.1%}",
                    "Alpha":           f"{m.get('alpha', 0):+.1%}",
                    "Beta":            f"{m.get('beta', 0):.2f}",
                    "Info Ratio":      f"{m.get('information_ratio', 0):.2f}",
                    "Roll. 12M Alpha": f"{m.get('rolling_12m_alpha', 0):+.1%}",
                    "vs SPY":          f"{(m.get('annual_return', 0) - m.get('benchmark_return', 0)):+.1%}",
                })
            except Exception:
                metrics_rows.append({"Strategy": strategy_labels[strategy], "Ann. Return": "N/A"})

        if metrics_rows:
            st.dataframe(pd.DataFrame(metrics_rows), hide_index=True)
    except Exception as e:
        st.error(f"Metrics error: {e}")

    # ── Drawdown Chart ────────────────────────────────────────────────────────
    if selected_strategies:
        st.subheader("Drawdown")
        try:
            from indexes.performance import compute_index_returns, compute_drawdown_series
            fig_dd = go.Figure()
            for strategy in selected_strategies:
                idx = compute_index_returns(f"{strategy}_index", start_str, end_str)
                if not idx.empty:
                    ret = idx.pct_change().dropna()
                    dd  = compute_drawdown_series(ret) * 100
                    fig_dd.add_trace(go.Scatter(
                        x=dd.index, y=dd.values,
                        name=strategy_labels[strategy],
                        fill="tozeroy", mode="lines",
                    ))
            fig_dd.update_layout(yaxis_title="Drawdown (%)", height=300, hovermode="x unified")
            st.plotly_chart(fig_dd, width='stretch')
        except Exception as e:
            st.warning(f"Drawdown chart unavailable: {e}")

    # ── Alpha Decay Tracker ───────────────────────────────────────────────────
    st.subheader("Rolling 12-Month Alpha (Alpha Decay Check)")
    st.info(
        "If rolling alpha is trending toward zero, the factor edge may be degrading. "
        "Review and consider running the reweighting optimizer."
    )
    # Placeholder chart — would show rolling alpha as a time series
    st.caption("Rolling alpha chart requires ≥12 months of index history.")
