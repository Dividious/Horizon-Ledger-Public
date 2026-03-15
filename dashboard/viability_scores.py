"""Horizon Ledger — Stock Viability Scores Dashboard Page"""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DISCLAIMER
from db.schema import get_connection
from db.queries import get_latest_scores
from newsletter.sections import get_viability_explanation, FACTOR_LABELS


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_components(raw: str) -> dict:
    """Parse score_components JSON string → dict, returning {} on failure."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _key_strengths(components: dict) -> str:
    """Return top-2 factor labels for display (delegates to sections helper)."""
    return get_viability_explanation(components)


def _build_display_df(scores_df: pd.DataFrame, min_score: float) -> pd.DataFrame:
    """Convert raw scores DataFrame into a display-ready DataFrame."""
    if scores_df.empty:
        return pd.DataFrame()

    df = scores_df.copy().reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    # Apply score filter
    df = df[df["composite_score"] >= min_score].copy()

    # Parse components
    df["_components"] = df["score_components"].apply(_parse_components)
    df["Key Strengths"] = df["_components"].apply(_key_strengths)

    display = df[["rank", "ticker", "name", "composite_score", "Key Strengths", "sector"]].copy()
    display.columns = ["Rank", "Ticker", "Company", "Score", "Key Strengths", "Sector"]
    display["Score"] = display["Score"].round(1)
    return display, df  # return both: display version and full df (with _components)


# ─── Factor breakdown radar chart ────────────────────────────────────────────

def _render_factor_breakdown(ticker: str, components: dict, strategy: str) -> None:
    """Render a radar / bar chart of factor percentiles for a selected ticker."""
    if not components:
        st.info(f"No factor data available for {ticker}.")
        return

    # Filter to known factor labels only
    factors = []
    values  = []
    for key, val in components.items():
        label = FACTOR_LABELS.get(key, key.replace("_pct", "").replace("_", " ").title())
        try:
            factors.append(label)
            values.append(float(val))
        except (TypeError, ValueError):
            pass

    if not factors:
        st.info(f"No recognizable factor data for {ticker}.")
        return

    # Bar chart — simpler and more readable in Streamlit than radar
    factor_df = pd.DataFrame({"Factor": factors, "Percentile": values}).sort_values(
        "Percentile", ascending=True
    )
    fig = px.bar(
        factor_df,
        x="Percentile",
        y="Factor",
        orientation="h",
        title=f"{ticker} — Factor Percentiles ({strategy.title()} strategy)",
        range_x=[0, 100],
        color="Percentile",
        color_continuous_scale=["#dc3545", "#ffc107", "#28a745"],
        labels={"Percentile": "Percentile Score (0–100)", "Factor": ""},
        height=max(300, len(factors) * 28),
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=150, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    factor_df_disp = factor_df.sort_values("Percentile", ascending=False).reset_index(drop=True)
    factor_df_disp["Percentile"] = factor_df_disp["Percentile"].round(1)
    st.dataframe(factor_df_disp, hide_index=True, use_container_width=True)


# ─── Main page ────────────────────────────────────────────────────────────────

def show() -> None:
    st.title("🏆 Stock Viability Scores")
    st.caption(DISCLAIMER)
    st.markdown(
        "Viability scores rank all ~1,000 stocks using quantitative factors. "
        "Higher score = stronger signal. **NOT a buy recommendation.**"
    )

    conn = get_connection()

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        min_score = st.slider(
            "Minimum composite score filter:",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
        )
    with ctrl_col2:
        detail_strategy = st.selectbox(
            "Strategy for factor detail:",
            options=["conservative", "aggressive"],
            format_func=str.title,
        )

    # ── Load scores ───────────────────────────────────────────────────────────
    con_scores_raw = get_latest_scores(conn, "conservative")
    agg_scores_raw = get_latest_scores(conn, "aggressive")

    if con_scores_raw.empty and agg_scores_raw.empty:
        st.warning(
            "No viability scores found. "
            "Run the scoring pipeline (scripts/run_weekly.py) first."
        )
        conn.close()
        return

    # ── Side-by-side top 25 columns ───────────────────────────────────────────
    st.subheader("Top 25 Viability Scores")
    col_con, col_agg = st.columns(2)

    con_display = pd.DataFrame()
    agg_display = pd.DataFrame()
    con_full = pd.DataFrame()
    agg_full = pd.DataFrame()

    with col_con:
        st.markdown("#### 🛡️ Conservative")
        if not con_scores_raw.empty:
            result = _build_display_df(con_scores_raw, min_score)
            if isinstance(result, tuple):
                con_display, con_full = result
            if not con_display.empty:
                st.dataframe(
                    con_display.head(25),
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(f"{len(con_display)} stocks above score {min_score:.0f}")
            else:
                st.info(f"No stocks above score {min_score:.0f}.")
        else:
            st.info("Conservative scores not yet available.")

    with col_agg:
        st.markdown("#### ⚡ Aggressive")
        if not agg_scores_raw.empty:
            result = _build_display_df(agg_scores_raw, min_score)
            if isinstance(result, tuple):
                agg_display, agg_full = result
            if not agg_display.empty:
                st.dataframe(
                    agg_display.head(25),
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(f"{len(agg_display)} stocks above score {min_score:.0f}")
            else:
                st.info(f"No stocks above score {min_score:.0f}.")
        else:
            st.info("Aggressive scores not yet available.")

    st.divider()

    # ── Factor breakdown for a selected ticker ────────────────────────────────
    st.subheader("Factor Breakdown")
    st.caption(
        "Select a ticker to see its individual factor percentile scores for "
        "the chosen strategy."
    )

    # Build ticker list from the selected strategy
    if detail_strategy == "conservative":
        full_df = con_full if not con_full.empty else pd.DataFrame()
        raw_df  = con_scores_raw
    else:
        full_df = agg_full if not agg_full.empty else pd.DataFrame()
        raw_df  = agg_scores_raw

    # Collect available tickers (filtered by min_score)
    if not full_df.empty and "_components" in full_df.columns:
        avail_tickers = full_df["Ticker"].tolist() if "Ticker" in full_df.columns else []
    elif not raw_df.empty:
        filtered_raw = raw_df[raw_df["composite_score"] >= min_score]
        avail_tickers = filtered_raw["ticker"].tolist()
    else:
        avail_tickers = []

    if not avail_tickers:
        st.info("No tickers available for factor breakdown at the current score filter.")
        conn.close()
        return

    selected_ticker = st.selectbox(
        f"Select ticker ({detail_strategy.title()} strategy):",
        options=avail_tickers[:200],  # cap list length for UI performance
    )

    if selected_ticker:
        # Get components from full_df if available, otherwise re-parse raw
        components = {}
        if not full_df.empty and "_components" in full_df.columns:
            ticker_mask = full_df.get("Ticker", pd.Series(dtype=str)) == selected_ticker
            if not ticker_mask.empty and ticker_mask.any():
                components = full_df.loc[ticker_mask, "_components"].iloc[0]
        if not components and not raw_df.empty:
            mask = raw_df["ticker"] == selected_ticker
            if mask.any():
                raw_comp = raw_df.loc[mask, "score_components"].iloc[0]
                components = _parse_components(raw_comp)

        with st.expander(
            f"Factor Percentiles for {selected_ticker} ({detail_strategy.title()})",
            expanded=True,
        ):
            _render_factor_breakdown(selected_ticker, components, detail_strategy)

    conn.close()
