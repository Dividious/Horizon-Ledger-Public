"""
Horizon Ledger — Streamlit Entry Point
Run with: streamlit run main.py

NOT FINANCIAL ADVICE. Personal research tool only.
"""

import sys
from pathlib import Path

# Add project root to path so all imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from config import DISCLAIMER

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Horizon Ledger",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ensure DB is initialized ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _init():
    from db.schema import init_db, seed_initial_weights, get_connection
    init_db()
    conn = get_connection()
    seed_initial_weights(conn)
    conn.close()

_init()

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Horizon Ledger")
    st.caption("Personal Investment Research System")
    st.divider()

    page = st.radio(
        "Navigation",
        options=[
            "🌐 Market Digest",
            "📊 Portfolio Overview",
            "🔍 Screening Results",
            "⚖️ Rebalancing",
            "📉 Performance",
            "🔎 Stock Detail",
            "⚙️ Reweighting",
            "🧪 Portfolio Simulator",
            "📈 Public Indexes",
            "🏆 Viability Scores",
            "📰 Newsletter",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("**Quick Actions**")
    if st.button("🔄 Refresh Data", use_container_width=True, help="Fetch latest prices"):
        with st.spinner("Fetching prices..."):
            try:
                from pipeline.prices import update_daily_prices
                update_daily_prices()
                st.success("Prices updated!")
            except Exception as e:
                st.error(f"Update failed: {e}")

    if st.button("📊 Run Scoring", use_container_width=True):
        with st.spinner("Running all scoring models..."):
            try:
                from datetime import date
                today = date.today().isoformat()
                from scoring.long_term import score_universe as lt_score
                from scoring.dividend  import score_universe as dv_score
                from scoring.turnaround import score_universe as ta_score
                from scoring.swing import score_universe as sw_score
                lt_score(as_of=today)
                dv_score(as_of=today)
                ta_score(as_of=today)
                sw_score(as_of=today)
                st.success("Scoring complete!")
            except Exception as e:
                st.error(f"Scoring failed: {e}")

    if st.button("🌐 Update Digest", use_container_width=True, help="Refresh market health score & digest"):
        with st.spinner("Updating market digest..."):
            try:
                from datetime import date
                from pipeline.market_health import update_market_digest
                digest = update_market_digest(as_of=date.today().isoformat())
                st.success(
                    f"Digest updated! Score: {digest.get('market_health_score', 0)} "
                    f"({digest.get('market_health_label', 'unknown')})"
                )
            except Exception as e:
                st.error(f"Digest update failed: {e}")

    st.divider()
    st.warning(DISCLAIMER, icon="⚠️")

# ── Page Routing ──────────────────────────────────────────────────────────────
if "Market Digest" in page:
    from dashboard.digest import show
    show()

elif "Portfolio Overview" in page:
    from dashboard.overview import show
    show()

elif "Screening Results" in page:
    from dashboard.screening import show
    show()

elif "Rebalancing" in page:
    from dashboard.rebalancing import show
    show()

elif "Performance" in page:
    from dashboard.performance import show
    show()

elif "Stock Detail" in page:
    from dashboard.stock_detail import show
    show()

elif "Reweighting" in page:
    from dashboard.reweighting_ui import show
    show()

elif "Portfolio Simulator" in page:
    from dashboard.simulator import show
    show()

elif "Public Indexes" in page:
    from dashboard.public_indexes import show
    show()

elif "Viability Scores" in page:
    from dashboard.viability_scores import show
    show()

elif "Newsletter" in page:
    from dashboard.newsletter_preview import show
    show()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **NOT FINANCIAL ADVICE.** Horizon Ledger is a personal research tool. "
    "All outputs are for informational purposes only. Data from yfinance, SEC EDGAR, and FRED. "
    "Past performance does not guarantee future results."
)
