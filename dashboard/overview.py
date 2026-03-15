"""
Horizon Ledger — Portfolio Overview Page
Shows index cards, holdings, sector allocation, sector rotation,
correlation matrix, top movers, and data freshness monitor.
"""

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import INDEX_SIZES, DISCLAIMER
from db.schema import get_connection
from db.queries import get_current_holdings, get_current_regime


def show():
    st.title("📊 Portfolio Overview")
    st.caption(DISCLAIMER)

    conn = get_connection()

    # ── Macro Regime Card ─────────────────────────────────────────────────────
    regime_row = get_current_regime(conn)
    if regime_row:
        regime = regime_row["regime"]
        conf   = max(regime_row["prob_bear"], regime_row["prob_neutral"], regime_row["prob_bull"])
        regime_color = {"bull": "🟢", "neutral": "🟡", "bear": "🔴"}.get(regime, "⚪")
        st.info(
            f"{regime_color} **Macro Regime:** {regime.upper()}  —  "
            f"Confidence: {conf:.0%}  |  "
            f"Bear {regime_row['prob_bear']:.0%} / Neutral {regime_row['prob_neutral']:.0%} / Bull {regime_row['prob_bull']:.0%}"
        )
    else:
        st.warning("Macro regime not yet computed. Run scripts/run_weekly.py first.")

    # ── Strategy Selector ─────────────────────────────────────────────────────
    strategies = ["long_term", "dividend", "turnaround", "swing"]
    strategy_labels = {
        "long_term":  "📈 Long-Term Buy & Hold",
        "dividend":   "💰 Dividend / Income",
        "turnaround": "🔄 High-Risk Turnaround",
        "swing":      "⚡ Short-Term Swing",
    }

    # ── Index Summary Cards ───────────────────────────────────────────────────
    cols = st.columns(len(strategies))
    for col, strategy in zip(cols, strategies):
        index_name = f"{strategy}_index"
        holdings = get_current_holdings(conn, index_name)
        n_holdings = len(holdings)

        try:
            from indexes.performance import get_all_metrics
            metrics = get_all_metrics(index_name, start=(date.today() - timedelta(days=365)).isoformat())
            annual_ret = metrics.get("annual_return", 0)
            sharpe     = metrics.get("sharpe_ratio", 0)
            ret_str    = f"{annual_ret:+.1%}"
            sharpe_str = f"{sharpe:.2f}"
        except Exception:
            ret_str    = "N/A"
            sharpe_str = "N/A"

        with col:
            st.metric(
                label=strategy_labels[strategy],
                value=f"{n_holdings} holdings",
                delta=f"Return: {ret_str}",
            )
            st.caption(f"Sharpe: {sharpe_str}")

    st.divider()

    # ── Selected Index Deep-Dive ──────────────────────────────────────────────
    selected = st.selectbox(
        "Select index for details:",
        options=strategies,
        format_func=lambda s: strategy_labels[s],
    )
    index_name = f"{selected}_index"
    holdings_df = get_current_holdings(conn, index_name)

    if holdings_df.empty:
        st.warning(f"No holdings in {index_name} yet. Run reconstitution first.")
        conn.close()
        _show_data_freshness()
        return

    # Add return since entry
    holdings_df["return_since_entry"] = _compute_entry_returns(holdings_df, conn)

    # ── Holdings Table ────────────────────────────────────────────────────────
    st.subheader(f"{strategy_labels[selected]} — Current Holdings")
    display_cols = ["ticker", "name", "sector", "target_weight", "entry_date", "return_since_entry", "entry_score"]
    display_df = holdings_df[[c for c in display_cols if c in holdings_df.columns]].copy()
    display_df["target_weight"] = (display_df["target_weight"] * 100).round(1).astype(str) + "%"
    if "return_since_entry" in display_df.columns:
        display_df["return_since_entry"] = display_df["return_since_entry"].apply(
            lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A"
        )
    if "entry_score" in display_df.columns:
        display_df["entry_score"] = display_df["entry_score"].round(1)

    st.dataframe(display_df, hide_index=True)

    # ── Two-column layout: Sector Pie + Correlation Matrix ────────────────────
    col_sector, col_corr = st.columns(2)

    with col_sector:
        if "sector" in holdings_df.columns:
            sector_weights = holdings_df.groupby("sector")["target_weight"].sum().reset_index()
            sector_weights.columns = ["sector", "weight"]
            fig = px.pie(
                sector_weights, values="weight", names="sector",
                title=f"Sector Allocation — {index_name}", hole=0.3,
            )
            st.plotly_chart(fig, width='stretch')

    with col_corr:
        st.subheader("Holdings Correlation Matrix")
        st.caption("1-year return correlations. High values = hidden concentration risk.")
        _show_correlation_matrix(holdings_df, conn)

    # ── Sector Rotation Chart ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Sector Rotation Tracker")
    st.caption("Sector ETF return minus SPY return. Green = outperforming, Red = underperforming.")
    _show_sector_rotation()

    # ── Top 5 Movers ──────────────────────────────────────────────────────────
    st.subheader("Top Movers This Week")
    movers = _compute_weekly_movers(holdings_df, conn)
    if not movers.empty:
        movers_display = movers.head(10).copy()
        movers_display["week_return"] = movers_display["week_return"].apply(
            lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A"
        )
        st.dataframe(movers_display[["ticker", "sector", "week_return"]], hide_index=True)

    conn.close()

    # ── Data Freshness Monitor ─────────────────────────────────────────────────
    st.divider()
    _show_data_freshness()


# ─── Helper functions ─────────────────────────────────────────────────────────

def _show_correlation_matrix(holdings_df: pd.DataFrame, conn):
    """Render a correlation heatmap for the index holdings returns."""
    try:
        from db.queries import get_prices
        tickers = holdings_df["ticker"].tolist()[:20]
        start = (date.today() - timedelta(days=365)).isoformat()
        returns_dict = {}

        for _, h_row in holdings_df[holdings_df["ticker"].isin(tickers)].iterrows():
            tkr = h_row["ticker"]
            sid = int(h_row["stock_id"])
            prices = get_prices(conn, sid, start=start)
            if not prices.empty and len(prices) > 20:
                prices = prices.sort_values("date").set_index("date")["adj_close"]
                returns_dict[tkr] = prices.pct_change().dropna()

        if len(returns_dict) < 3:
            st.info("Not enough price history for correlation matrix.")
            return

        ret_df = pd.DataFrame(returns_dict).dropna(how="all")
        corr_matrix = ret_df.corr()

        fig = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdYlGn",
            zmin=-1, zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 9},
            hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
        ))
        fig.update_layout(
            height=max(300, len(tickers) * 22),
            xaxis_tickangle=-45,
            margin=dict(t=10, b=60, l=60, r=10),
        )
        st.plotly_chart(fig, width='stretch')
    except Exception as e:
        st.caption(f"Correlation matrix unavailable: {e}")


def _show_sector_rotation():
    """Show sector ETF relative strength vs SPY as a heatmap."""
    sector_etfs = {
        "XLK": "Technology",  "XLV": "Health Care",  "XLF": "Financials",
        "XLY": "Consumer Disc.", "XLI": "Industrials", "XLC": "Communication",
        "XLP": "Consumer Stap.", "XLE": "Energy",      "XLB": "Materials",
        "XLRE": "Real Estate", "XLU": "Utilities",
    }
    timeframes = {"1M": 21, "3M": 63, "6M": 126}
    try:
        conn = get_connection()
        today = date.today().isoformat()
        from db.queries import get_stock_id, get_price_on_date
        spy_id  = get_stock_id(conn, "SPY")
        spy_now = get_price_on_date(conn, spy_id, today) if spy_id else None

        rows = []
        for ticker, sector_name in sector_etfs.items():
            sid   = get_stock_id(conn, ticker)
            if not sid:
                continue
            p_now = get_price_on_date(conn, sid, today)
            if not p_now:
                continue
            row = {"Sector": sector_name, "ETF": ticker}
            for label, days in timeframes.items():
                start     = (date.today() - timedelta(days=days)).isoformat()
                p_start   = get_price_on_date(conn, sid, start)
                spy_start = get_price_on_date(conn, spy_id, start) if spy_id else None
                if p_start and p_start > 0:
                    etf_ret = (p_now - p_start) / p_start
                    if spy_start and spy_now and spy_start > 0:
                        row[label] = etf_ret - (spy_now - spy_start) / spy_start
                    else:
                        row[label] = etf_ret
                else:
                    row[label] = None
            rows.append(row)
        conn.close()

        if not rows:
            st.info("Sector ETF data not yet available. Run scripts/run_daily.py.")
            return

        df = pd.DataFrame(rows)
        numeric_cols = list(timeframes.keys())
        heat_data = df.set_index("Sector")[numeric_cols]
        fig = go.Figure(go.Heatmap(
            z=heat_data.values,
            x=heat_data.columns.tolist(),
            y=heat_data.index.tolist(),
            colorscale="RdYlGn", zmid=0,
            text=[[f"{v:+.1%}" if pd.notna(v) else "N/A" for v in row]
                  for row in heat_data.values],
            texttemplate="%{text}",
            hovertemplate="%{y} (%{x}): %{text}<extra></extra>",
        ))
        fig.update_layout(
            height=320, title="Sector Relative Return vs SPY",
            margin=dict(t=40, b=10, l=120, r=20),
        )
        st.plotly_chart(fig, width='stretch')
    except Exception as e:
        st.caption(f"Sector rotation unavailable: {e}")


def _show_data_freshness():
    """Show last-updated timestamps for all major data sources."""
    st.subheader("🔍 Data Freshness Monitor")
    st.caption("🟢 Fresh (<2d)  🟡 Aging (2-7d)  🔴 Stale (>7d)  ⚪ Never updated")
    try:
        conn = get_connection()
        today = date.today()

        def _days_since(date_str):
            if not date_str:
                return None
            try:
                return (today - date.fromisoformat(str(date_str)[:10])).days
            except Exception:
                return None

        def _staleness(days):
            if days is None:
                return "⚪ Never"
            if days <= 2:
                return f"🟢 {days}d ago"
            elif days <= 7:
                return f"🟡 {days}d ago"
            return f"🔴 {days}d ago"

        data = {
            "Daily Prices":    _days_since(conn.execute("SELECT MAX(date) FROM daily_prices").fetchone()[0]),
            "Fundamentals":    _days_since(conn.execute("SELECT MAX(filing_date) FROM fundamentals").fetchone()[0]),
            "FRED Macro":      _days_since(conn.execute("SELECT MAX(date) FROM macro_data").fetchone()[0]),
            "Scores":          _days_since(conn.execute("SELECT MAX(score_date) FROM scores").fetchone()[0]),
            "HMM Regime":      _days_since(conn.execute("SELECT MAX(date) FROM regime_history").fetchone()[0]),
            "Market Digest":   _days_since(conn.execute("SELECT MAX(date) FROM market_digest_history").fetchone()[0]),
        }
        conn.close()

        scripts = {
            "Daily Prices": "run_daily.py",
            "FRED Macro": "run_daily.py",
            "Market Digest": "run_daily.py",
            "Fundamentals": "run_weekly.py",
            "Scores": "run_weekly.py",
            "HMM Regime": "run_weekly.py",
        }

        cols = st.columns(len(data))
        for col, (label, days) in zip(cols, data.items()):
            with col:
                st.metric(label, _staleness(days))
                if days is not None and days > 7:
                    st.caption(f"→ `{scripts.get(label, '')}`")
    except Exception as e:
        st.caption(f"Freshness monitor unavailable: {e}")


def _compute_entry_returns(holdings_df: pd.DataFrame, conn) -> pd.Series:
    from db.queries import get_price_on_date
    returns = []
    today = date.today().isoformat()
    for _, h in holdings_df.iterrows():
        entry_p = h.get("entry_price")
        curr_p  = get_price_on_date(conn, h["stock_id"], today)
        if entry_p and curr_p and entry_p > 0:
            returns.append((curr_p - entry_p) / entry_p)
        else:
            returns.append(None)
    return pd.Series(returns, index=holdings_df.index)


def _compute_weekly_movers(holdings_df: pd.DataFrame, conn) -> pd.DataFrame:
    from db.queries import get_price_on_date
    today    = date.today().isoformat()
    week_ago = (date.today() - timedelta(days=5)).isoformat()
    rows = []
    for _, h in holdings_df.iterrows():
        p_now  = get_price_on_date(conn, h["stock_id"], today)
        p_week = get_price_on_date(conn, h["stock_id"], week_ago)
        ret = (p_now - p_week) / p_week if (p_now and p_week and p_week > 0) else None
        rows.append({"ticker": h["ticker"], "sector": h.get("sector"), "week_return": ret})
    df = pd.DataFrame(rows)
    if not df.empty and "week_return" in df.columns:
        df = df.sort_values("week_return", ascending=False)
    return df
