"""
Horizon Ledger — Individual Stock Deep-Dive Page
Candlestick chart, technical indicators, factor scores, Altman Z, Piotroski F, Beneish M.
"""

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import DISCLAIMER, ALTMAN_SAFE_ZONE, ALTMAN_GREY_ZONE, ALTMAN_DISTRESS, BENEISH_THRESHOLD
from db.schema import get_connection
from db.queries import (
    get_stock_id, get_prices, get_latest_fundamentals, get_latest_scores,
    add_stock_note, get_stock_notes, delete_stock_note,
)


def show():
    st.title("🔎 Stock Detail")
    st.caption(DISCLAIMER)

    conn = get_connection()

    # ── Ticker search ─────────────────────────────────────────────────────────
    ticker_input = st.text_input("Ticker Symbol:", placeholder="e.g. AAPL").upper().strip()
    if not ticker_input:
        st.info("Enter a ticker symbol to view details.")
        conn.close()
        return

    sid = get_stock_id(conn, ticker_input)
    if sid is None:
        st.error(f"Ticker '{ticker_input}' not found in universe. Add it via universe.py.")
        conn.close()
        return

    # Stock info
    stock_row = conn.execute("SELECT * FROM stocks WHERE id=?", (sid,)).fetchone()
    if stock_row:
        s = dict(stock_row)
        st.subheader(f"{s.get('name', ticker_input)} ({ticker_input})")
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Sector",   s.get("sector", "N/A"))
        colB.metric("Industry", s.get("industry", "N/A") or "N/A")
        colC.metric("Exchange", s.get("exchange", "N/A") or "N/A")
        mcap = s.get("market_cap")
        colD.metric("Market Cap", f"${mcap/1e9:.1f}B" if mcap else "N/A")

    # ── Date range for chart ──────────────────────────────────────────────────
    end_date  = date.today()
    start_date = end_date - timedelta(days=365)

    start_str = start_date.isoformat()
    end_str   = end_date.isoformat()

    # ── Price data ────────────────────────────────────────────────────────────
    prices_df = get_prices(conn, sid, start=start_str, end=end_str)
    if prices_df.empty:
        st.warning("No price data found. Run prices.py backfill first.")
        conn.close()
        return

    # ── Technical indicators ──────────────────────────────────────────────────
    try:
        from pipeline.technicals import compute_indicators
        tech_df = compute_indicators(prices_df)
    except Exception as e:
        tech_df = prices_df.copy()
        st.caption(f"Technical indicators unavailable: {e}")

    # ── Candlestick + Technical Chart ─────────────────────────────────────────
    st.subheader("Price Chart")
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.05,
        subplot_titles=["Price + MAs + BB", "RSI (14)", "MACD"],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=prices_df["date"],
        open=prices_df["open"],
        high=prices_df["high"],
        low=prices_df["low"],
        close=prices_df["adj_close"],
        name="OHLC",
        showlegend=False,
    ), row=1, col=1)

    # SMA 50 & 200
    if "sma_50" in tech_df.columns:
        fig.add_trace(go.Scatter(x=tech_df["date"], y=tech_df["sma_50"],
                                  name="SMA 50", line=dict(color="blue", width=1)), row=1, col=1)
    if "sma_200" in tech_df.columns:
        fig.add_trace(go.Scatter(x=tech_df["date"], y=tech_df["sma_200"],
                                  name="SMA 200", line=dict(color="orange", width=1.5)), row=1, col=1)

    # Bollinger Bands
    if "bb_upper" in tech_df.columns:
        fig.add_trace(go.Scatter(x=tech_df["date"], y=tech_df["bb_upper"],
                                  line=dict(color="gray", dash="dot", width=1), name="BB Upper", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=tech_df["date"], y=tech_df["bb_lower"],
                                  line=dict(color="gray", dash="dot", width=1), name="BB Lower", fill="tonexty",
                                  fillcolor="rgba(100,100,100,0.1)", showlegend=False), row=1, col=1)

    # Volume
    if "volume" in prices_df.columns:
        fig.add_trace(go.Bar(x=prices_df["date"], y=prices_df["volume"],
                              name="Volume", marker_color="lightblue", showlegend=False), row=1, col=1)

    # RSI
    if "rsi_14" in tech_df.columns:
        fig.add_trace(go.Scatter(x=tech_df["date"], y=tech_df["rsi_14"],
                                  name="RSI 14", line=dict(color="purple")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    if "macd_line" in tech_df.columns:
        fig.add_trace(go.Scatter(x=tech_df["date"], y=tech_df["macd_line"],
                                  name="MACD", line=dict(color="blue")), row=3, col=1)
        fig.add_trace(go.Scatter(x=tech_df["date"], y=tech_df["macd_signal"],
                                  name="Signal", line=dict(color="orange")), row=3, col=1)
        if "macd_histogram" in tech_df.columns:
            colors = ["green" if v >= 0 else "red" for v in tech_df["macd_histogram"].fillna(0)]
            fig.add_trace(go.Bar(x=tech_df["date"], y=tech_df["macd_histogram"],
                                  marker_color=colors, name="Histogram", showlegend=False), row=3, col=1)

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    st.plotly_chart(fig, width='stretch')

    st.divider()

    # ── Factor Score Radar ────────────────────────────────────────────────────
    st.subheader("Factor Scores by Strategy")
    strategies = ["long_term", "dividend", "turnaround", "swing"]
    from db.queries import get_latest_scores as gls

    all_scores_df = gls(conn, "long_term")  # We'll query each
    radar_data: dict[str, dict] = {}

    for strategy in strategies:
        rows = conn.execute(
            """SELECT score_components FROM scores
               WHERE stock_id=? AND strategy=?
               ORDER BY score_date DESC LIMIT 1""",
            (sid, strategy),
        ).fetchone()
        if rows and rows["score_components"]:
            try:
                comp = json.loads(rows["score_components"])
                radar_data[strategy] = comp
            except Exception:
                pass

    if radar_data:
        strategy_labels_short = {
            "long_term":  "Long-Term",
            "dividend":   "Dividend",
            "turnaround": "Turnaround",
            "swing":      "Swing",
        }
        radar_cols = st.columns(len(radar_data))
        for col, (strategy, comps) in zip(radar_cols, radar_data.items()):
            with col:
                st.caption(f"**{strategy_labels_short[strategy]}**")
                comp_df = pd.DataFrame([{"Factor": k, "Score": v} for k, v in comps.items()]).sort_values("Score", ascending=False)
                st.dataframe(comp_df, hide_index=True)

    st.divider()

    # ── Fundamental Snapshot ──────────────────────────────────────────────────
    st.subheader("Fundamental Snapshot")
    fund = get_latest_fundamentals(conn, sid)
    if fund is not None:
        mcap = (stock_row or {}).get("market_cap") if stock_row else None
        mcap_val = mcap if mcap else 0

        col1, col2, col3, col4 = st.columns(4)
        def _fmt(v, fmt=".2f", suffix=""):
            return f"{v:{fmt}}{suffix}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "N/A"

        col1.metric("Revenue",   _fmt(fund.get("revenue"), ".2e"))
        col1.metric("Gross Profit", _fmt(fund.get("gross_profit"), ".2e"))
        col2.metric("EBIT",      _fmt(fund.get("ebit"), ".2e"))
        col2.metric("Net Income",_fmt(fund.get("net_income"), ".2e"))
        col3.metric("Total Assets", _fmt(fund.get("total_assets"), ".2e"))
        col3.metric("Total Debt",   _fmt(fund.get("total_debt"), ".2e"))
        col4.metric("Free Cash Flow", _fmt(fund.get("free_cash_flow"), ".2e"))
        col4.metric("Shares Out",    _fmt(fund.get("shares_outstanding"), ".2e"))

    st.divider()

    # ── Altman Z-Score Gauge ──────────────────────────────────────────────────
    st.subheader("Altman Z-Score")
    if fund is not None:
        from scoring.composite import altman_z_from_row
        mcap_val = (dict(stock_row).get("market_cap") or 0) if stock_row else 0
        z = altman_z_from_row(fund, market_cap=mcap_val)
        if z is not None:
            if z >= ALTMAN_SAFE_ZONE:
                color, zone = "green", "Safe Zone"
            elif z >= ALTMAN_GREY_ZONE:
                color, zone = "orange", "Grey Zone"
            else:
                color, zone = "red", "Distress Zone"

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=z,
                title={"text": f"Altman Z-Score ({zone})"},
                gauge={
                    "axis": {"range": [-2, 10]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [-2, ALTMAN_DISTRESS], "color": "lightcoral"},
                        {"range": [ALTMAN_DISTRESS, ALTMAN_SAFE_ZONE], "color": "lightyellow"},
                        {"range": [ALTMAN_SAFE_ZONE, 10], "color": "lightgreen"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": z},
                },
            ))
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, width='stretch')
        else:
            st.info("Altman Z-Score unavailable (insufficient data).")

    # ── Piotroski F-Score ─────────────────────────────────────────────────────
    st.subheader("Piotroski F-Score")
    if fund is not None:
        from scoring.composite import piotroski_f_score
        fdf_2q = conn.execute(
            """SELECT * FROM fundamentals WHERE stock_id=?
               ORDER BY report_date DESC LIMIT 5""", (sid,)
        ).fetchall()
        if fdf_2q:
            curr = dict(fdf_2q[0])
            prev = dict(fdf_2q[4]) if len(fdf_2q) > 4 else None
            f_score = piotroski_f_score(curr, prev)
            f_color = "🟢" if f_score >= 7 else ("🟡" if f_score >= 5 else "🔴")
            st.metric("F-Score", f"{f_color} {f_score}/9",
                       delta="Strong" if f_score >= 7 else ("Moderate" if f_score >= 5 else "Weak"))

    # ── Beneish M-Score ───────────────────────────────────────────────────────
    st.subheader("Beneish M-Score")
    fdf_all = conn.execute(
        "SELECT * FROM fundamentals WHERE stock_id=? ORDER BY report_date DESC LIMIT 2",
        (sid,),
    ).fetchall()
    if len(fdf_all) >= 2:
        from scoring.composite import beneish_m_score
        m = beneish_m_score(dict(fdf_all[0]), dict(fdf_all[1]))
        if m is not None:
            manipulated = m > BENEISH_THRESHOLD
            m_color = "🔴" if manipulated else "🟢"
            st.metric(
                "M-Score",
                f"{m_color} {m:.3f}",
                delta="⚠️ Likely manipulation" if manipulated else "✅ Within normal range",
                delta_color="off",
            )
            st.caption(f"Threshold: {BENEISH_THRESHOLD}. M > {BENEISH_THRESHOLD} indicates potential earnings manipulation.")
    else:
        st.info("Need 2+ quarters of data for Beneish M-Score.")

    # ── 52-Week Range ─────────────────────────────────────────────────────────
    st.subheader("52-Week Range")
    _show_52w_range(prices_df, ticker_input)

    # ── Upcoming Earnings ─────────────────────────────────────────────────────
    st.subheader("Upcoming Earnings")
    _show_upcoming_earnings(ticker_input)

    # ── Signal History ────────────────────────────────────────────────────────
    st.subheader("Score History")
    strategies_tabs = st.tabs(["Long-Term", "Dividend", "Turnaround", "Swing"])
    for tab, strategy in zip(strategies_tabs, strategies):
        with tab:
            hist = pd.read_sql(
                "SELECT score_date, composite_score FROM scores WHERE stock_id=? AND strategy=? ORDER BY score_date",
                conn, params=[sid, strategy],
            )
            if hist.empty:
                st.info("No score history yet.")
            else:
                fig_h = go.Figure(go.Scatter(x=hist["score_date"], y=hist["composite_score"],
                                              mode="lines+markers", name="Composite Score"))
                fig_h.update_layout(height=250, yaxis_title="Score (0-100)")
                st.plotly_chart(fig_h, width='stretch')

    # ── Personal Notes / Investment Thesis ────────────────────────────────────
    st.divider()
    st.subheader("📝 Personal Notes")
    _show_notes_section(conn, sid, ticker_input)

    conn.close()


def _show_52w_range(prices_df: pd.DataFrame, ticker: str):
    """Display 52-week price range bar."""
    try:
        from datetime import timedelta
        one_year_ago = (date.today() - timedelta(days=365)).isoformat()
        p52 = prices_df[prices_df["date"] >= one_year_ago]["adj_close"].dropna()
        if p52.empty:
            st.info("Insufficient price history for 52-week range.")
            return

        low_52   = float(p52.min())
        high_52  = float(p52.max())
        current  = float(prices_df["adj_close"].dropna().iloc[-1])
        pct_pos  = (current - low_52) / (high_52 - low_52) * 100 if high_52 > low_52 else 50.0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("52W Low",    f"${low_52:.2f}")
        col2.metric("Current",    f"${current:.2f}", delta=f"{(current/low_52-1):+.1%} from low")
        col3.metric("52W High",   f"${high_52:.2f}")
        col4.metric("Range Pos.", f"{pct_pos:.0f}%",
                    delta="Near high ⚠️" if pct_pos > 85
                    else ("Near low 💡" if pct_pos < 15 else "Mid-range"))

        # Visual range bar
        fig = go.Figure(go.Bar(
            x=[pct_pos], y=[ticker],
            orientation="h",
            marker_color="#3498db",
            width=0.4,
        ))
        fig.add_shape(type="line", x0=0, x1=100, y0=-0.5, y1=0.5,
                      line=dict(color="lightgray", width=2))
        fig.update_layout(
            height=80, xaxis=dict(range=[0, 100], ticksuffix="%"),
            yaxis=dict(visible=False), margin=dict(t=5, b=5, l=10, r=10),
            title=f"{ticker} position in 52-week range",
        )
        st.plotly_chart(fig, width='stretch')
    except Exception as e:
        st.caption(f"52-week range unavailable: {e}")


def _show_upcoming_earnings(ticker: str):
    """Show next earnings date from yfinance."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker)
        cal = info.calendar
        if cal is not None and not cal.empty:
            if isinstance(cal, dict):
                earn_dates = cal.get("Earnings Date", [])
            else:
                earn_dates = cal.get("Earnings Date", pd.Series()).tolist() if "Earnings Date" in cal else []

            if earn_dates:
                next_date = earn_dates[0] if earn_dates else None
                if next_date:
                    days_away = (pd.Timestamp(next_date).date() - date.today()).days
                    if days_away >= 0:
                        st.info(
                            f"📅 Next earnings: **{pd.Timestamp(next_date).strftime('%Y-%m-%d')}** "
                            f"({days_away} days away)"
                        )
                    else:
                        st.caption(f"Last reported earnings: {pd.Timestamp(next_date).strftime('%Y-%m-%d')}")
                    return

        st.caption("Earnings date not available from yfinance.")
    except Exception as e:
        st.caption(f"Earnings calendar unavailable: {e}")


def _show_notes_section(conn, stock_id: int, ticker: str):
    """Render the personal notes/thesis section for a stock."""
    notes_df = get_stock_notes(conn, stock_id)

    # Add new note form
    with st.expander("✏️ Add Note / Investment Thesis", expanded=False):
        note_type = st.selectbox(
            "Note type:",
            ["general", "thesis", "risk", "update"],
            format_func=lambda x: x.title(),
            key=f"note_type_{ticker}",
        )
        note_content = st.text_area(
            "Note:",
            placeholder=f"Why are you watching {ticker}? What's the thesis?",
            key=f"note_content_{ticker}",
            height=100,
        )
        if st.button("💾 Save Note", key=f"save_note_{ticker}"):
            if note_content.strip():
                with conn:
                    add_stock_note(conn, stock_id, note_content.strip(), note_type)
                    conn.commit()
                st.success("Note saved!")
                st.rerun()
            else:
                st.warning("Note is empty.")

    # Display existing notes
    if notes_df.empty:
        st.caption("No notes yet for this stock.")
    else:
        type_icons = {"thesis": "💡", "risk": "⚠️", "update": "🔄", "general": "📝"}
        for _, note in notes_df.iterrows():
            icon = type_icons.get(note.get("note_type", "general"), "📝")
            with st.container():
                col_note, col_del = st.columns([9, 1])
                with col_note:
                    st.markdown(
                        f"**{icon} {note['note_type'].title()}** — "
                        f"*{note['note_date']}*  \n{note['content']}"
                    )
                with col_del:
                    if st.button("🗑️", key=f"del_note_{note['id']}", help="Delete note"):
                        with conn:
                            delete_stock_note(conn, int(note["id"]))
                            conn.commit()
                        st.rerun()
                st.divider()
