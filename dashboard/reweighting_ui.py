"""
Horizon Ledger — Reweighting Review and Approval UI
Human-in-the-loop review of reweighting proposals with guardrail validation.
"""

import json
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    DISCLAIMER,
    REWEIGHTING_MAX_CHANGE_PER_CYCLE,
    REWEIGHTING_MAX_SINGLE_FACTOR,
    REWEIGHTING_MIN_SINGLE_FACTOR,
    STRATEGY_WEIGHTS,
)
from db.schema import get_connection
from db.queries import get_active_weights, get_latest_weight_version


def show():
    st.title("⚙️ Factor Reweighting")
    st.caption(DISCLAIMER)
    st.info(
        "This page lets you review, modify, and approve reweighting proposals. "
        "All weight changes are bounded by guardrails. Human approval is required "
        "unless changes are very small (< 2%) and all confidence intervals are positive."
    )

    conn = get_connection()
    strategies = ["long_term", "dividend", "turnaround", "swing"]
    strategy_labels = {
        "long_term":  "Long-Term Buy & Hold",
        "dividend":   "Dividend / Income",
        "turnaround": "High-Risk Turnaround",
        "swing":      "Short-Term Swing",
    }

    selected = st.selectbox(
        "Strategy:",
        options=strategies,
        format_func=lambda s: strategy_labels[s],
    )

    # ── Current Weights ───────────────────────────────────────────────────────
    st.subheader(f"Current Weights — {strategy_labels[selected]}")
    current_weights = get_active_weights(conn, selected)
    current_version = get_latest_weight_version(conn, selected)

    col_chart, col_meta = st.columns([3, 1])
    with col_chart:
        wdf = pd.DataFrame([{"Factor": f, "Weight": w * 100} for f, w in current_weights.items()])
        fig = px.bar(
            wdf, x="Weight", y="Factor", orientation="h",
            title="Current Factor Weights (%)",
            color_discrete_sequence=["#4e79a7"],
        )
        fig.update_layout(height=max(300, len(wdf) * 25), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width='stretch')
    with col_meta:
        st.metric("Current Version", current_version or "v1.0")
        total_w = sum(current_weights.values())
        st.metric("Weight Sum", f"{total_w:.3f}", delta="✅ OK" if abs(total_w - 1.0) < 0.001 else "⚠️ ERROR")

    # ── Pending Proposals ─────────────────────────────────────────────────────
    st.subheader("Pending Reweighting Proposals")
    from reweighting.proposal import get_pending_proposals, approve_proposal, reject_proposal

    proposals = get_pending_proposals(selected)
    if not proposals:
        st.info("No pending proposals. Generate one below or wait for the quarterly run.")
    else:
        for proposal in proposals:
            _render_proposal(conn, proposal, current_weights, selected)

    st.divider()

    # ── Factor Efficacy Scorecard ─────────────────────────────────────────────
    with st.expander("📊 Factor Efficacy Scorecard", expanded=False):
        _show_factor_scorecard(conn, selected, current_weights)

    st.divider()

    # ── Generate New Proposal ─────────────────────────────────────────────────
    st.subheader("Generate New Proposal")
    from reweighting.tracker import get_accuracy_summary
    accuracy = get_accuracy_summary(selected)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Predictions", accuracy["total_predictions"])
    col_b.metric("With 63d Returns", accuracy["filled_63d"])
    col_c.metric("Ready for Reweighting", "✅ Yes" if accuracy["sufficient_for_reweighting"] else f"❌ No — est. {accuracy['estimated_date_sufficient']}")

    if st.button("🔄 Generate Reweighting Proposal", disabled=not accuracy["sufficient_for_reweighting"]):
        with st.spinner("Running IC + ElasticNet optimization..."):
            try:
                from reweighting.proposal import generate_proposal
                result = generate_proposal(selected, save_to_db=True)
                st.success(f"Proposal generated! Recommendation: **{result.get('recommendation', 'N/A')}**")
                st.rerun()
            except Exception as e:
                st.error(f"Proposal generation failed: {e}")

    # ── Weight Version History ─────────────────────────────────────────────────
    st.subheader("Weight Version History")
    hist = pd.read_sql(
        "SELECT version_id, created_date, approved_by, notes FROM weight_versions WHERE strategy=? ORDER BY created_date DESC LIMIT 10",
        conn, params=[selected],
    )
    if hist.empty:
        st.info("No weight version history yet.")
    else:
        st.dataframe(hist, hide_index=True)

    conn.close()


def _show_factor_scorecard(conn, strategy: str, current_weights: dict):
    """
    Render a factor efficacy scorecard showing IC trend, t-stat, hit rate,
    and a 🟢/🟡/🔴 status per factor for the selected strategy.
    """
    try:
        from reweighting.tracker import compute_ic_statistics
        ic_stats = compute_ic_statistics(strategy)
        if not ic_stats:
            st.info(
                "No IC statistics available yet. "
                "Factors need forward returns filled before efficacy can be computed "
                "(requires several months of predictions)."
            )
            return

        rows = []
        for factor, stats in ic_stats.items():
            ic_12m  = stats.get("ic_12m")
            ic_6m   = stats.get("ic_6m")
            ic_ir   = stats.get("ic_ir")
            hit_rate = stats.get("hit_rate")
            t_stat  = stats.get("t_stat")

            # Status light
            if ic_12m is None:
                status = "⚪ No data"
            elif ic_12m >= 0.05 and (t_stat is None or t_stat >= 3.0):
                status = "🟢 Strong"
            elif ic_12m >= 0.01:
                status = "🟡 Weak"
            elif ic_12m < 0:
                status = "🔴 Negative"
            else:
                status = "🟡 Below min"

            rows.append({
                "Factor":         factor,
                "Weight":         f"{current_weights.get(factor, 0) * 100:.1f}%",
                "IC (12m)":       f"{ic_12m:.3f}" if ic_12m is not None else "N/A",
                "IC (6m)":        f"{ic_6m:.3f}"  if ic_6m  is not None else "N/A",
                "IC IR":          f"{ic_ir:.2f}"  if ic_ir  is not None else "N/A",
                "Hit Rate":       f"{hit_rate:.0%}" if hit_rate is not None else "N/A",
                "t-stat":         f"{t_stat:.2f}" if t_stat is not None else "N/A",
                "Status":         status,
            })

        if not rows:
            st.info("No factor efficacy data yet.")
            return

        df = pd.DataFrame(rows)
        st.markdown(
            "**Status legend:** 🟢 IC ≥ 0.05 & t-stat ≥ 3.0  |  "
            "🟡 IC 0.01–0.05 or t-stat below threshold  |  "
            "🔴 Negative IC — consider reviewing or removing"
        )
        st.dataframe(df, hide_index=True)

        # IC trend sparkline bars
        ic_vals = [float(r["IC (12m)"].replace("N/A", "nan")) if r["IC (12m)"] != "N/A" else None
                   for r in rows]
        factors = [r["Factor"] for r in rows]
        colors  = ["green" if (v and v >= 0.05) else
                   ("orange" if (v and v >= 0.01) else
                    ("red" if (v and v < 0) else "lightgray"))
                   for v in ic_vals]

        valid_indices = [(f, v, c) for f, v, c in zip(factors, ic_vals, colors) if v is not None]
        if valid_indices:
            f_vals, v_vals, c_vals = zip(*valid_indices)
            fig = go.Figure(go.Bar(
                x=list(f_vals), y=list(v_vals),
                marker_color=list(c_vals), name="IC (12m)",
            ))
            fig.add_hline(y=0.01, line_dash="dot", line_color="orange",
                          annotation_text="IC min (0.01)")
            fig.add_hline(y=0.05, line_dash="dot", line_color="green",
                          annotation_text="IC strong (0.05)")
            fig.update_layout(
                height=280, title="12-Month IC by Factor",
                xaxis_tickangle=-45, yaxis_title="IC",
            )
            st.plotly_chart(fig, width='stretch')

    except Exception as e:
        st.caption(f"Factor scorecard unavailable: {e}")


def _render_proposal(conn, proposal: dict, current_weights: dict, strategy: str):
    """Render a single proposal with comparison table, charts, and approve/reject UI."""
    from reweighting.proposal import approve_proposal, reject_proposal

    proposal_id  = proposal["id"]
    prop_date    = proposal.get("proposal_date", "")
    prop_weights = proposal.get("proposed_weights", {})
    rec          = proposal.get("recommendation", "REVIEW")
    wfe          = proposal.get("walk_forward_efficiency")
    oos_r2       = proposal.get("en_oos_r2")

    rec_color = {"APPROVE": "success", "MODIFY": "warning", "INVESTIGATE": "error"}.get(rec, "info")

    with st.expander(f"Proposal #{proposal_id} — {prop_date} — Recommendation: {rec}", expanded=True):

        # Recommendation banner
        getattr(st, rec_color)(
            f"**Recommendation: {rec}** — {proposal.get('recommendation_reason', '')}"
        )

        # Stats row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("WFE (OOS/IS)", f"{wfe:.3f}" if wfe else "N/A",
                     delta="⚠️ BELOW MIN" if (wfe and wfe < 0.5) else "")
        col2.metric("OOS R²", f"{oos_r2:.3f}" if oos_r2 else "N/A")
        col3.metric("Max Change", f"{proposal.get('max_change', 0):.1%}")
        col4.metric("Est. Turnover", f"{proposal.get('estimated_turnover_impact', 0):.1%}")

        # Side-by-side comparison table
        st.markdown("**Weight Comparison:**")
        all_factors = set(current_weights.keys()) | set(prop_weights.keys())
        comp_rows = []
        for f in sorted(all_factors):
            curr_w = current_weights.get(f, 0)
            prop_w = prop_weights.get(f, 0)
            change = prop_w - curr_w
            ci = (proposal.get("confidence_intervals") or {}).get(f, {})
            ci_str = f"[{ci.get('lower', 0):.3f}, {ci.get('upper', 0):.3f}]" if ci else "N/A"
            comp_rows.append({
                "Factor":         f,
                "Current (%)":    f"{curr_w * 100:.1f}",
                "Proposed (%)":   f"{prop_w * 100:.1f}",
                "Change (pp)":    f"{change * 100:+.1f}",
                "95% CI":         ci_str,
                "CI Crosses 0":   "⚠️" if ci.get("ci_crosses_zero") else "✅",
            })
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True)

        # Weight change bar chart
        if prop_weights:
            fig = go.Figure()
            factors = sorted(all_factors)
            curr_vals = [current_weights.get(f, 0) * 100 for f in factors]
            prop_vals = [prop_weights.get(f, 0) * 100 for f in factors]
            fig.add_trace(go.Bar(name="Current", x=factors, y=curr_vals, marker_color="#4e79a7"))
            fig.add_trace(go.Bar(name="Proposed", x=factors, y=prop_vals, marker_color="#f28e2b"))
            fig.update_layout(
                barmode="group", height=350, yaxis_title="Weight (%)",
                title="Current vs Proposed Weights",
            )
            st.plotly_chart(fig, width='stretch')

        # Flagged factors
        flags = []
        if proposal.get("flagged_negative_ic"):
            flags.append(f"⚠️ Negative IC: {proposal['flagged_negative_ic']}")
        if proposal.get("flagged_low_ic"):
            flags.append(f"⚠️ Low IC (< 0.01): {proposal['flagged_low_ic']}")
        if proposal.get("low_t_stat_factors"):
            flags.append(f"⚠️ Low t-stat (< 3.0): {proposal['low_t_stat_factors']}")
        for flag in flags:
            st.warning(flag)

        # ── Modify mode ───────────────────────────────────────────────────────
        st.markdown("**Manual Weight Adjustment (optional):**")
        with st.form(key=f"modify_weights_{proposal_id}"):
            modified_weights = {}
            cols_per_row = 3
            factor_list = list(current_weights.keys())
            for i in range(0, len(factor_list), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for j, f in enumerate(factor_list[i:i + cols_per_row]):
                    default_w = float(prop_weights.get(f, 0)) * 100
                    modified_weights[f] = row_cols[j].slider(
                        f,
                        min_value=0.0, max_value=40.0,
                        value=round(default_w, 1),
                        step=0.5,
                        key=f"slider_{proposal_id}_{f}",
                    ) / 100

            # Real-time validation
            total_modified = sum(modified_weights.values())
            weight_ok = abs(total_modified - 1.0) < 0.05
            st.metric("Sum of weights:", f"{total_modified:.3f}", delta="✅ OK" if weight_ok else "⚠️ Must sum to 1.0")

            guardrail_violations = [
                f for f, w in modified_weights.items()
                if w > 0 and (w > REWEIGHTING_MAX_SINGLE_FACTOR or w < REWEIGHTING_MIN_SINGLE_FACTOR)
            ]
            if guardrail_violations:
                st.warning(f"Guardrail violation — factor(s) outside allowed range: {guardrail_violations}")

            # Approve / Reject buttons
            col_ap, col_rj, col_mod = st.columns(3)
            approve_btn   = col_ap.form_submit_button("✅ Approve Proposed",  type="primary")
            reject_btn    = col_rj.form_submit_button("❌ Reject",             type="secondary")
            modify_btn    = col_mod.form_submit_button("💾 Approve Modified",  type="secondary")

            if approve_btn:
                if wfe is not None and wfe < 0.5:
                    st.error("Cannot approve: Walk-Forward Efficiency below minimum (0.5). Investigate first.")
                else:
                    # Normalize
                    total = sum(prop_weights.values())
                    norm_weights = {f: v / total for f, v in prop_weights.items()} if total > 0 else prop_weights
                    approve_proposal(proposal_id, norm_weights, approved_by="human")
                    st.success("Proposal approved! New weights are active.")
                    st.rerun()

            if reject_btn:
                reject_proposal(proposal_id, reason="Rejected via dashboard")
                st.info("Proposal rejected.")
                st.rerun()

            if modify_btn:
                if not weight_ok:
                    st.error("Cannot approve: weights must sum to ~1.0")
                elif guardrail_violations:
                    st.error(f"Cannot approve: guardrail violations for {guardrail_violations}")
                else:
                    norm = sum(modified_weights.values())
                    final = {f: v / norm for f, v in modified_weights.items()}
                    approve_proposal(proposal_id, final, approved_by="human_modified")
                    st.success("Modified weights approved!")
                    st.rerun()
