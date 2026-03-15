"""
Horizon Ledger — Screening Results Page
Full ranked universe table with color coding, filters, and export.
"""

import json
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from config import INDEX_SIZES, DISCLAIMER
from db.schema import get_connection
from db.queries import get_latest_scores, get_current_holdings, get_prices, get_stock_id


def show():
    st.title("🔍 Screening Results")
    st.caption(DISCLAIMER)

    strategy_labels = {
        "long_term":  "📈 Long-Term Buy & Hold",
        "dividend":   "💰 Dividend / Income",
        "turnaround": "🔄 High-Risk Turnaround",
        "swing":      "⚡ Short-Term Swing",
    }

    # ── Strategy selector ─────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        strategy = st.selectbox(
            "Strategy:",
            options=list(strategy_labels.keys()),
            format_func=lambda s: strategy_labels[s],
        )
    with col2:
        show_excluded = st.checkbox("Show excluded stocks", value=False)

    conn = get_connection()
    scores_df = get_latest_scores(conn, strategy)

    if scores_df.empty:
        st.warning(f"No scores found for '{strategy}'. Run the scoring pipeline first.")
        conn.close()
        return

    # Merge with current holdings for "In Index?" flag
    index_name = f"{strategy}_index"
    holdings   = get_current_holdings(conn, index_name)
    conn.close()

    in_index_tickers = set(holdings["ticker"].tolist()) if not holdings.empty else set()
    target_size = INDEX_SIZES.get(strategy, 25)
    exit_buffer_rank = int(target_size * 1.4)

    # Parse score components
    scores_df["score_components_parsed"] = scores_df["score_components"].apply(
        lambda x: json.loads(x) if x else {}
    )

    # Rank column
    scores_df = scores_df.reset_index(drop=True)
    scores_df["rank"] = range(1, len(scores_df) + 1)

    scores_df["in_index"] = scores_df["ticker"].isin(in_index_tickers)
    scores_df["status"] = scores_df.apply(
        lambda r: "IN INDEX" if r["in_index"]
        else ("BORDERLINE" if r["rank"] <= exit_buffer_rank else "OUT"),
        axis=1,
    )

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("Filters", expanded=False):
        fcol1, fcol2, fcol3 = st.columns(3)

        sectors = ["All"] + sorted(scores_df["sector"].dropna().unique().tolist())
        with fcol1:
            sel_sector = st.selectbox("Sector:", sectors)

        with fcol2:
            score_range = st.slider(
                "Composite score range:",
                min_value=0.0, max_value=100.0,
                value=(0.0, 100.0), step=1.0,
            )
        with fcol3:
            status_filter = st.multiselect(
                "Status:",
                options=["IN INDEX", "BORDERLINE", "OUT"],
                default=["IN INDEX", "BORDERLINE", "OUT"],
            )

    # Apply filters
    filtered = scores_df.copy()
    if sel_sector != "All":
        filtered = filtered[filtered["sector"] == sel_sector]
    filtered = filtered[
        (filtered["composite_score"] >= score_range[0]) &
        (filtered["composite_score"] <= score_range[1])
    ]
    if status_filter:
        filtered = filtered[filtered["status"].isin(status_filter)]

    # ── Results table ─────────────────────────────────────────────────────────
    st.subheader(f"{len(filtered)} stocks shown ({len(scores_df)} total)")

    # ── Add 52-week range context ────────────────────────────────────────────
    filtered = _add_52w_range(filtered, conn)

    display_cols = ["rank", "ticker", "name", "sector", "composite_score",
                    "status", "in_index", "52w_position", "52w_low", "52w_high"]
    display_df = filtered[[c for c in display_cols if c in filtered.columns]].copy()
    display_df["composite_score"] = display_df["composite_score"].round(1)

    def _style_row(row):
        # Always include color:#000 so text stays readable on dark/light themes.
        if row.get("status") == "IN INDEX":
            return ["background-color: #d4edda; color: #1a1a1a"] * len(row)
        elif row.get("status") == "BORDERLINE":
            return ["background-color: #fff3cd; color: #1a1a1a"] * len(row)
        elif row.get("status") == "OUT":
            return ["background-color: #2c2c2c; color: #cccccc"] * len(row)
        return ["background-color: #2c2c2c; color: #cccccc"] * len(row)

    st.dataframe(
        display_df.style.apply(_style_row, axis=1),
        hide_index=True,
    )

    # ── Factor score expandable ───────────────────────────────────────────────
    with st.expander("Factor Scores for Selected Stocks"):
        selected_tickers = st.multiselect(
            "Select tickers:", options=filtered["ticker"].tolist(), max_selections=5
        )
        if selected_tickers:
            sel_df = filtered[filtered["ticker"].isin(selected_tickers)].copy()
            factor_rows = []
            for _, row in sel_df.iterrows():
                comp = row.get("score_components_parsed", {})
                factor_rows.append({"ticker": row["ticker"], **comp})
            st.dataframe(pd.DataFrame(factor_rows), hide_index=True)

    # ── Score distribution histogram ──────────────────────────────────────────
    col_main, col_side = st.columns([3, 1])
    with col_side:
        st.subheader("Score Distribution")
        fig = px.histogram(
            scores_df,
            x="composite_score",
            nbins=20,
            color="status",
            color_discrete_map={
                "IN INDEX": "#28a745",
                "BORDERLINE": "#ffc107",
                "OUT": "#6c757d",
            },
            title="",
        )
        fig.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig, width='stretch')

    # ── Export ────────────────────────────────────────────────────────────────
    conn.close()
    csv = filtered.drop(columns=["score_components_parsed", "score_components"], errors="ignore").to_csv(index=False)
    st.download_button(
        label="📥 Export to CSV",
        data=csv,
        file_name=f"screening_{strategy}_{date.today().isoformat()}.csv",
        mime="text/csv",
    )


def _add_52w_range(scores_df: pd.DataFrame, conn) -> pd.DataFrame:
    """
    Add 52-week high/low context and position (% of range) for each stock.
    Uses the daily_prices table directly — fast batch computation.
    """
    try:
        from datetime import timedelta
        start = (date.today() - timedelta(days=365)).isoformat()
        today = date.today().isoformat()

        # Batch query for all stocks in the scored universe
        tickers = scores_df["ticker"].tolist()
        if not tickers:
            return scores_df

        placeholders = ",".join(["?"] * len(tickers))
        df_prices = pd.read_sql(
            f"""
            SELECT s.ticker, dp.date, dp.adj_close
            FROM daily_prices dp
            JOIN stocks s ON s.id = dp.stock_id
            WHERE s.ticker IN ({placeholders})
              AND dp.date >= ? AND dp.date <= ?
            ORDER BY dp.date
            """,
            conn, params=tickers + [start, today],
        )

        if df_prices.empty:
            return scores_df

        # Compute 52w stats per ticker
        stats = df_prices.groupby("ticker")["adj_close"].agg(
            low_52w="min", high_52w="max", current="last"
        ).reset_index()
        stats.columns = ["ticker", "52w_low", "52w_high", "current_price"]
        stats["52w_position"] = stats.apply(
            lambda r: f"{(r['current_price'] - r['52w_low']) / (r['52w_high'] - r['52w_low']) * 100:.0f}%"
            if (r["52w_high"] - r["52w_low"]) > 0 else "N/A",
            axis=1,
        )
        stats["52w_low"]  = stats["52w_low"].round(2)
        stats["52w_high"] = stats["52w_high"].round(2)
        scores_df = scores_df.merge(
            stats[["ticker", "52w_low", "52w_high", "52w_position"]],
            on="ticker", how="left",
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("52w range failed: %s", e)

    return scores_df
