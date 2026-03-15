"""
Horizon Ledger — Rebalancing Dashboard Page
Shows drift analysis, pending proposals, and rebalancing history.
"""

import json
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import REBALANCE_DRIFT_THRESHOLD, DISCLAIMER
from db.schema import get_connection


def show():
    st.title("⚖️ Rebalancing")
    st.caption(DISCLAIMER)

    strategies = ["long_term", "dividend", "turnaround", "swing"]
    strategy_labels = {
        "long_term":  "Long-Term Buy & Hold",
        "dividend":   "Dividend / Income",
        "turnaround": "High-Risk Turnaround",
        "swing":      "Short-Term Swing",
    }

    selected_strategy = st.selectbox(
        "Strategy:",
        options=strategies,
        format_func=lambda s: strategy_labels[s],
    )
    index_name = f"{selected_strategy}_index"

    # ── Drift Analysis ────────────────────────────────────────────────────────
    st.subheader(f"Weight Drift — {strategy_labels[selected_strategy]}")
    try:
        from indexes.rebalancer import check_drift, generate_rebalancing_proposal
        drift_df = check_drift(index_name)
    except Exception as e:
        st.error(f"Error computing drift: {e}")
        drift_df = pd.DataFrame()

    if drift_df.empty:
        st.info("No holdings yet in this index.")
    else:
        # Drift bar chart
        fig = go.Figure()
        colors = ["red" if v else "green" for v in drift_df["exceeds_threshold"]]
        fig.add_trace(go.Bar(
            x=drift_df["ticker"],
            y=drift_df["drift"] * 100,
            marker_color=colors,
            name="Drift (%)",
        ))
        fig.add_hline(
            y=REBALANCE_DRIFT_THRESHOLD * 100,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Threshold ({REBALANCE_DRIFT_THRESHOLD:.0%})",
        )
        fig.update_layout(
            title="Weight Drift from Target",
            xaxis_title="Ticker",
            yaxis_title="Drift (%)",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Drift table
        drift_display = drift_df.copy()
        for col in ["target_weight", "current_weight", "drift"]:
            if col in drift_display.columns:
                drift_display[col] = (drift_display[col] * 100).round(2).astype(str) + "%"
        st.dataframe(drift_display, hide_index=True)

        n_flagged = drift_df["exceeds_threshold"].sum()
        if n_flagged > 0:
            st.warning(f"⚠️ {n_flagged} position(s) exceed the {REBALANCE_DRIFT_THRESHOLD:.0%} drift threshold.")
        else:
            st.success("✅ All positions within drift tolerance.")

    # ── Reconstitution Proposal ───────────────────────────────────────────────
    # Proposals are stored in session_state keyed by strategy so they survive
    # rerenders when the user switches strategies or clicks Apply.
    st.subheader("Generate Rebalancing Proposal")

    proposal_key = f"proposal_{selected_strategy}"

    if st.button("🔄 Run Reconstitution (Dry Run)", key=f"dry_run_{selected_strategy}"):
        try:
            from indexes.builder import reconstitute_index
            proposal = reconstitute_index(selected_strategy, dry_run=True)
            st.session_state[proposal_key] = proposal
        except Exception as e:
            st.error(f"Reconstitution error: {e}")
            st.session_state.pop(proposal_key, None)

    # Show proposal results if one exists in session state for this strategy
    if proposal_key in st.session_state:
        proposal = st.session_state[proposal_key]
        adds    = proposal.get("adds", [])
        removes = proposal.get("removes", [])
        wc      = proposal.get("weight_changes", [])

        colA, colB, colC = st.columns(3)
        colA.metric("Additions", len(adds))
        colB.metric("Removals", len(removes))
        colC.metric("Weight Changes", len(wc))

        if adds:
            st.markdown("**Proposed Additions:**")
            add_df = pd.DataFrame([{
                "ticker": a["ticker"],
                "target_weight": f"{a['target_weight']:.1%}",
                "entry_score": round(a.get("entry_score") or 0, 1),
            } for a in adds])
            st.dataframe(add_df, hide_index=True)

        if removes:
            st.markdown("**Proposed Removals:**")
            rem_df = pd.DataFrame([{
                "ticker": r["ticker"],
                "reason": r.get("reason", ""),
            } for r in removes])
            st.dataframe(rem_df, hide_index=True)

        if wc:
            st.markdown("**Weight Adjustments:**")
            wc_df = pd.DataFrame([{
                "ticker": w["ticker"],
                "old_weight": f"{w['old_weight']:.1%}",
                "new_weight": f"{w['new_weight']:.1%}",
            } for w in wc])
            st.dataframe(wc_df, hide_index=True)

        # Apply button is always visible once a proposal exists — no nesting issue
        st.divider()
        col_apply, col_clear = st.columns([2, 1])
        with col_apply:
            if st.button(
                "✅ Apply Reconstitution",
                key=f"apply_recon_{selected_strategy}",
                type="primary",
            ):
                with st.spinner("Applying reconstitution..."):
                    try:
                        from indexes.builder import reconstitute_index
                        reconstitute_index(selected_strategy, dry_run=False)
                        st.session_state.pop(proposal_key, None)
                        st.success(
                            f"✅ {strategy_labels[selected_strategy]} reconstitution applied! "
                            f"{len(adds)} additions, {len(removes)} removals."
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Apply failed: {e}")
        with col_clear:
            if st.button("🗑️ Clear Proposal", key=f"clear_{selected_strategy}"):
                st.session_state.pop(proposal_key, None)
                st.rerun()

    # ── Rebalancing History ───────────────────────────────────────────────────
    st.subheader("Rebalancing History")
    conn = get_connection()
    hist_df = pd.read_sql(
        """SELECT rh.rebalance_date, rh.action, s.ticker, rh.old_weight, rh.new_weight, rh.reason
           FROM rebalancing_history rh
           LEFT JOIN stocks s ON s.id = rh.stock_id
           WHERE rh.index_name=?
           ORDER BY rh.rebalance_date DESC
           LIMIT 100""",
        conn,
        params=[index_name],
    )
    conn.close()

    if hist_df.empty:
        st.info("No rebalancing history yet.")
    else:
        hist_df["old_weight"] = (hist_df["old_weight"] * 100).round(2).astype(str) + "%"
        hist_df["new_weight"] = (hist_df["new_weight"] * 100).round(2).astype(str) + "%"
        st.dataframe(hist_df, hide_index=True)
