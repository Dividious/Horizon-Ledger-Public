"""
Horizon Ledger — Bubble / Sector Valuation Flags
Computes risk flags from CAPE, sector valuations, credit spreads, and yield curve.

Flags indicate ELEVATED RISK, not a sell signal.  All flags include context
(historical percentile, base rates) to avoid false precision.

NOT FINANCIAL ADVICE.
"""

import json
import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    CAPE_STRETCHED_THRESHOLD,
    CAPE_EXTREME_THRESHOLD,
    SECTOR_VALUATION_STRETCH_MULTIPLIER,
)
from db.schema import get_connection
from db.queries import get_macro_series

log = logging.getLogger(__name__)


# ─── CAPE Flag ────────────────────────────────────────────────────────────────

def _cape_flag(
    cape_ratio: Optional[float],
    cape_percentile: Optional[float],
) -> dict:
    """
    Returns a flag dict if CAPE is above thresholds, with historical context.
    At CAPE > 30: historical 10y forward returns have averaged ~4% annualized.
    At CAPE > 35: historical 10y forward returns have averaged ~1-2% annualized.
    """
    flags = {}
    if cape_ratio is None:
        return flags

    pct_str = f"{cape_percentile:.0f}th percentile" if cape_percentile else "unknown percentile"

    # Approximate expected return lookup (from Shiller historical data)
    if cape_ratio > 30:
        exp_return = "~1-2%" if cape_ratio > 35 else "~4%"
        key = "market_extreme" if cape_ratio > CAPE_EXTREME_THRESHOLD else "market_stretched"
        flags[key] = (
            f"CAPE at {cape_ratio:.1f} ({pct_str} historically). "
            f"10-year forward returns from this level have averaged {exp_return} annualized."
        )

    return flags


# ─── Sector Valuation Flags ───────────────────────────────────────────────────

def _sector_valuation_flags() -> dict:
    """
    For each GICS sector, compare median P/E, P/S, EV/EBITDA to their 5-year
    historical medians from the fundamentals table.
    Flag a sector if 2+ metrics are > SECTOR_VALUATION_STRETCH_MULTIPLIER × median.
    Returns dict: {sector: "stretched" | "elevated" | "normal" | "cheap"}
    """
    flags = {}
    try:
        conn = get_connection()
        today = date.today().isoformat()
        five_yr_ago = (date.today() - timedelta(days=365 * 5)).isoformat()

        # Get latest fundamentals + prices + sectors for all active stocks
        df = pd.read_sql(
            """
            SELECT st.sector,
                   f.revenue, f.net_income, f.total_assets, f.total_debt,
                   f.operating_cash_flow, f.gross_profit, f.ebit,
                   f.shares_outstanding, f.filing_date,
                   st.market_cap
            FROM fundamentals f
            JOIN stocks st ON st.id = f.stock_id
            JOIN (
                SELECT stock_id, MAX(filing_date) as max_fd
                FROM fundamentals
                WHERE filing_date <= ?
                GROUP BY stock_id
            ) latest ON latest.stock_id = f.stock_id AND f.filing_date = latest.max_fd
            WHERE st.is_active = 1 AND st.sector IS NOT NULL
            """,
            conn,
            params=[today],
        )

        if df.empty:
            conn.close()
            return {}

        # Compute approximate valuation ratios (best effort from available data)
        df["mcap"] = pd.to_numeric(df["market_cap"], errors="coerce").fillna(0)
        df["rev"]  = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)
        df["ni"]   = pd.to_numeric(df["net_income"], errors="coerce").fillna(0)
        df["ebit"] = pd.to_numeric(df["ebit"], errors="coerce").fillna(0)
        df["debt"] = pd.to_numeric(df["total_debt"], errors="coerce").fillna(0)
        df["cash_flow"] = pd.to_numeric(df["operating_cash_flow"], errors="coerce").fillna(0)

        # Simple EV = market_cap + total_debt (approximation)
        df["ev"] = df["mcap"] + df["debt"]

        df["pe"]       = np.where((df["ni"] > 0) & (df["mcap"] > 0), df["mcap"] / df["ni"], np.nan)
        df["ps"]       = np.where((df["rev"] > 0) & (df["mcap"] > 0), df["mcap"] / df["rev"], np.nan)
        df["ev_ebitda"] = np.where(
            (df["ebit"] > 0) & (df["ev"] > 0), df["ev"] / df["ebit"], np.nan
        )

        # Clip extreme outliers (> 200 PE, > 50 PS, > 100 EV/EBITDA)
        df["pe"]        = df["pe"].clip(0, 200)
        df["ps"]        = df["ps"].clip(0, 50)
        df["ev_ebitda"] = df["ev_ebitda"].clip(0, 100)

        # Get 5-year historical medians from historical fundamentals
        hist_df = pd.read_sql(
            """
            SELECT st.sector,
                   f.revenue, f.net_income, f.ebit, f.total_debt,
                   st.market_cap, f.filing_date
            FROM fundamentals f
            JOIN stocks st ON st.id = f.stock_id
            WHERE st.is_active = 1 AND st.sector IS NOT NULL
              AND f.filing_date BETWEEN ? AND ?
            """,
            conn,
            params=[five_yr_ago, today],
        )
        conn.close()

        if hist_df.empty:
            return {}

        hist_df["mcap"] = pd.to_numeric(hist_df["market_cap"], errors="coerce").fillna(0)
        hist_df["rev"]  = pd.to_numeric(hist_df["revenue"], errors="coerce").fillna(0)
        hist_df["ni"]   = pd.to_numeric(hist_df["net_income"], errors="coerce").fillna(0)
        hist_df["ebit"] = pd.to_numeric(hist_df["ebit"], errors="coerce").fillna(0)
        hist_df["debt"] = pd.to_numeric(hist_df["total_debt"], errors="coerce").fillna(0)

        hist_df["pe"]       = np.where((hist_df["ni"] > 0) & (hist_df["mcap"] > 0),
                                       hist_df["mcap"] / hist_df["ni"], np.nan).clip(0, 200)
        hist_df["ps"]       = np.where((hist_df["rev"] > 0) & (hist_df["mcap"] > 0),
                                       hist_df["mcap"] / hist_df["rev"], np.nan).clip(0, 50)
        ev_h = hist_df["mcap"] + hist_df["debt"]
        hist_df["ev_ebitda"] = np.where((hist_df["ebit"] > 0) & (ev_h > 0),
                                        ev_h / hist_df["ebit"], np.nan).clip(0, 100)

        historical_medians = hist_df.groupby("sector")[["pe", "ps", "ev_ebitda"]].median()
        current_medians    = df.groupby("sector")[["pe", "ps", "ev_ebitda"]].median()
        mult = SECTOR_VALUATION_STRETCH_MULTIPLIER

        for sector in current_medians.index:
            if sector not in historical_medians.index:
                continue
            h = historical_medians.loc[sector]
            c = current_medians.loc[sector]

            stretched_count = 0
            if pd.notna(c["pe"]) and pd.notna(h["pe"]) and h["pe"] > 0:
                if c["pe"] > h["pe"] * mult:
                    stretched_count += 1
            if pd.notna(c["ps"]) and pd.notna(h["ps"]) and h["ps"] > 0:
                if c["ps"] > h["ps"] * mult:
                    stretched_count += 1
            if pd.notna(c["ev_ebitda"]) and pd.notna(h["ev_ebitda"]) and h["ev_ebitda"] > 0:
                if c["ev_ebitda"] > h["ev_ebitda"] * mult:
                    stretched_count += 1

            if stretched_count >= 2:
                flags[sector] = "stretched"

    except Exception as e:
        log.warning("Sector valuation flags failed: %s", e)

    return flags


# ─── Credit Spread Flag ───────────────────────────────────────────────────────

def _credit_spread_flag(credit_spread: Optional[float]) -> dict:
    """
    Flag 'credit_complacency' if current spread is at/below 10th historical percentile.
    Note: BAMLH0A0HYM2 data starts 1997 in FRED.
    """
    flags = {}
    if credit_spread is None:
        return flags
    try:
        conn = get_connection()
        hist = get_macro_series(conn, "BAMLH0A0HYM2")
        conn.close()
        if hist.empty or len(hist) < 50:
            return flags
        p10 = hist["value"].quantile(0.10)
        if credit_spread <= p10:
            flags["credit_complacency"] = (
                f"HY spread at {credit_spread:.0f} bps — at or below the 10th "
                f"historical percentile ({p10:.0f} bps). Low spreads can signal market "
                "complacency toward credit risk (data since 1997)."
            )
    except Exception as e:
        log.warning("Credit spread flag failed: %s", e)
    return flags


# ─── Yield Curve Inversion Flag ───────────────────────────────────────────────

def _yield_curve_flag(yield_curve_inverted: bool) -> dict:
    """
    Flag 'inversion_sustained' if the yield curve has been inverted > 90 days.
    In the post-1960 US record, sustained inversions (> 90 days) have preceded
    8 of the last 8 recessions (with variable lead times of 6-24 months).
    """
    flags = {}
    if not yield_curve_inverted:
        return flags
    try:
        conn = get_connection()
        dgs10 = get_macro_series(conn, "DGS10",
                                 start=(date.today() - timedelta(days=730)).isoformat())
        dgs2  = get_macro_series(conn, "DGS2",
                                 start=(date.today() - timedelta(days=730)).isoformat())
        conn.close()
        if dgs10.empty or dgs2.empty:
            return flags
        merged = dgs10.rename(columns={"value": "dgs10"}).merge(
            dgs2.rename(columns={"value": "dgs2"}), on="date"
        )
        merged["slope"] = merged["dgs10"] - merged["dgs2"]
        merged = merged.sort_values("date", ascending=False)
        count = 0
        for _, r in merged.iterrows():
            if r["slope"] < 0:
                count += 1
            else:
                break
        if count > 90:
            flags["inversion_sustained"] = (
                f"Yield curve has been inverted for {count} consecutive days. "
                "In the post-1960 US record, inversions of this duration have "
                "preceded 8 of the last 8 recessions (lead time: 6-24 months)."
            )
    except Exception as e:
        log.warning("Yield curve inversion flag failed: %s", e)
    return flags


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def compute_bubble_flags(
    cape_ratio: Optional[float] = None,
    cape_percentile: Optional[float] = None,
    credit_spread: Optional[float] = None,
    yield_curve_inverted: bool = False,
) -> dict:
    """
    Compute all active bubble/risk flags.
    Returns a flat dict of {flag_key: explanation_string}.
    Store the result as JSON in market_digest_history.bubble_flags.
    """
    flags = {}
    flags.update(_cape_flag(cape_ratio, cape_percentile))
    flags.update(_credit_spread_flag(credit_spread))
    flags.update(_yield_curve_flag(yield_curve_inverted))
    flags.update(_sector_valuation_flags())
    return flags


def get_sector_valuation_detail() -> pd.DataFrame:
    """
    Return a DataFrame with per-sector current and 5-year median valuations.
    Used by the dashboard sector heatmap.
    Columns: sector, pe_current, pe_hist_median, ps_current, ps_hist_median,
             ev_ebitda_current, ev_ebitda_hist_median, stretched_count
    """
    try:
        conn = get_connection()
        today = date.today().isoformat()
        five_yr_ago = (date.today() - timedelta(days=365 * 5)).isoformat()

        current_df = pd.read_sql(
            """SELECT st.sector, f.revenue, f.net_income, f.ebit,
                      f.total_debt, st.market_cap
               FROM fundamentals f
               JOIN stocks st ON st.id = f.stock_id
               JOIN (
                   SELECT stock_id, MAX(filing_date) as max_fd
                   FROM fundamentals WHERE filing_date <= ? GROUP BY stock_id
               ) lat ON lat.stock_id = f.stock_id AND f.filing_date = lat.max_fd
               WHERE st.is_active=1 AND st.sector IS NOT NULL""",
            conn, params=[today],
        )
        hist_df = pd.read_sql(
            """SELECT st.sector, f.revenue, f.net_income, f.ebit,
                      f.total_debt, st.market_cap
               FROM fundamentals f
               JOIN stocks st ON st.id = f.stock_id
               WHERE st.is_active=1 AND st.sector IS NOT NULL
                 AND f.filing_date BETWEEN ? AND ?""",
            conn, params=[five_yr_ago, today],
        )
        conn.close()

        def _add_ratios(df):
            df = df.copy()
            df["mcap"] = pd.to_numeric(df["market_cap"], errors="coerce").fillna(0)
            df["rev"]  = pd.to_numeric(df["revenue"],    errors="coerce").fillna(0)
            df["ni"]   = pd.to_numeric(df["net_income"], errors="coerce").fillna(0)
            df["ebit"] = pd.to_numeric(df["ebit"],       errors="coerce").fillna(0)
            df["debt"] = pd.to_numeric(df["total_debt"], errors="coerce").fillna(0)
            df["pe"]       = np.where((df["ni"]>0)&(df["mcap"]>0), df["mcap"]/df["ni"], np.nan)
            df["ps"]       = np.where((df["rev"]>0)&(df["mcap"]>0), df["mcap"]/df["rev"], np.nan)
            df["ev_ebitda"] = np.where((df["ebit"]>0)&(df["mcap"]+df["debt"]>0),
                                       (df["mcap"]+df["debt"])/df["ebit"], np.nan)
            df["pe"]        = df["pe"].clip(0, 200)
            df["ps"]        = df["ps"].clip(0, 50)
            df["ev_ebitda"] = df["ev_ebitda"].clip(0, 100)
            return df

        curr_r = _add_ratios(current_df).groupby("sector")[["pe", "ps", "ev_ebitda"]].median()
        hist_r = _add_ratios(hist_df).groupby("sector")[["pe", "ps", "ev_ebitda"]].median()
        curr_r.columns = ["pe_current", "ps_current", "ev_ebitda_current"]
        hist_r.columns = ["pe_hist",    "ps_hist",    "ev_ebitda_hist"]

        combined = curr_r.join(hist_r, how="left").reset_index()
        mult = SECTOR_VALUATION_STRETCH_MULTIPLIER

        def _count_stretched(row):
            count = 0
            for metric in ["pe", "ps", "ev_ebitda"]:
                c = row.get(f"{metric}_current")
                h = row.get(f"{metric}_hist")
                if pd.notna(c) and pd.notna(h) and h > 0 and c > h * mult:
                    count += 1
            return count

        combined["stretched_count"] = combined.apply(_count_stretched, axis=1)
        combined["status"] = combined["stretched_count"].map(
            {0: "normal", 1: "elevated", 2: "stretched", 3: "stretched"}
        ).fillna("normal")

        return combined

    except Exception as e:
        log.warning("Sector valuation detail failed: %s", e)
        return pd.DataFrame()
