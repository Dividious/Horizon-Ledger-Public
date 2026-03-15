"""
Horizon Ledger — Conservative Index Scoring
Conservative Index — Quality + Value + Low-Vol + Income.
Target audience: retirement/long-term investors.

Factor weights (initial, from config.py — updated via reweighting system):
  gross_profitability, roic, piotroski_f, altman_z_norm,
  earnings_yield, fcf_yield, ev_to_ebitda_inv,
  low_vol_252d_inv, debt_to_equity_inv, current_ratio,
  dividend_yield, div_growth_5y, consecutive_increases

Risk overlay (post-ranking exclusions):
  - Exclude Altman Z < 1.81 (ALTMAN_DISTRESS)
  - Exclude 252-day annualized vol > 45%
  - Exclude market cap < $5B

Target: top 25 stocks. Rebalance quarterly.

References:
  Novy-Marx (2013): gross profitability — predictive power
  Piotroski (2000): F-Score value investing signal
  Altman (1968): Z-Score distress detection
  Baker & Haugen (2012): Low-volatility anomaly
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    CONSERVATIVE_WEIGHTS,
    ALTMAN_DISTRESS,
    MISSING_FACTOR_FILL,
    INITIAL_WEIGHT_VERSION,
    SECTOR_NEUTRALIZE,
)
from db.schema import get_connection
from db.queries import (
    get_active_universe,
    get_fundamentals_as_of,
    get_active_weights,
    get_latest_weight_version,
    upsert_score,
    upsert_prediction,
    get_prices,
)
from scoring.composite import (
    percentile_rank, weighted_composite_score,
    piotroski_f_score,
    altman_z_from_row, compute_enterprise_value,
    compute_cagr, safe_div, MISSING_FACTOR_FILL as _FILL,
)
from scoring.utils import sector_neutralize

log = logging.getLogger(__name__)

STRATEGY = "conservative"

# Hard exclusion thresholds
_MAX_VOL_252D   = 0.45        # 45% annualized volatility ceiling
_MIN_MARKET_CAP = 5_000_000_000  # $5B minimum market cap


def compute_raw_factors(
    ticker: str,
    stock_id: int,
    conn,
    as_of: str,
    market_cap: Optional[float] = None,
) -> dict:
    """
    Compute all raw (un-normalized) factor values for a single stock.
    Returns dict of {factor_name: value}.  None = data unavailable.
    """
    factors: dict[str, Optional[float]] = {}

    # Get fundamentals (point-in-time, up to 20 quarters for 5-year history)
    fdf = get_fundamentals_as_of(conn, stock_id, as_of, n_quarters=20)
    if fdf.empty:
        return factors

    curr   = fdf.iloc[0]                                                    # Most recent quarter
    prev_4 = fdf.iloc[4]  if len(fdf) > 4  else None                       # ~1 year ago
    prev_20 = fdf.iloc[-1] if len(fdf) >= 20 else (fdf.iloc[-1] if len(fdf) > 1 else None)

    ta     = curr.get("total_assets")
    eq     = curr.get("total_equity")
    td     = curr.get("total_debt")
    rev    = curr.get("revenue")
    gp     = curr.get("gross_profit")
    ebit   = curr.get("ebit")
    fcf    = curr.get("free_cash_flow")
    ca     = curr.get("current_assets")
    cl     = curr.get("current_liabilities")
    cash   = curr.get("cash")
    shares = curr.get("shares_outstanding")
    div    = curr.get("dividends_paid")   # Typically negative in EDGAR data (cash outflow)

    # Enterprise Value
    ev = compute_enterprise_value(market_cap, td, cash) if market_cap else None

    # ── Quality ───────────────────────────────────────────────────────────────
    # Gross Profitability = GP / Total Assets (Novy-Marx 2013)
    factors["gross_profitability"] = safe_div(gp, ta)

    # ROIC = EBIT / (Total Debt + Total Equity)
    invested_capital = (td or 0) + (eq or 0)
    factors["roic"] = safe_div(ebit, invested_capital if invested_capital != 0 else None)

    # Piotroski F-Score (raw 0–9; percentile applied cross-sectionally)
    factors["piotroski_f"] = float(piotroski_f_score(curr, prev_4))

    # Altman Z-Score (higher = safer; already "good = high" direction)
    factors["altman_z_norm"] = altman_z_from_row(curr, market_cap=market_cap)

    # ── Value ─────────────────────────────────────────────────────────────────
    # Earnings Yield = EBIT / EV
    factors["earnings_yield"] = safe_div(ebit, ev)

    # FCF Yield = FCF / EV
    factors["fcf_yield"] = safe_div(fcf, ev)

    # EV/EBITDA (inverted — lower multiple is better)
    ebitda_proxy = ebit or 0   # Simplified: EBIT used as EBITDA proxy
    ev_ebitda = safe_div(ev, ebitda_proxy if ebitda_proxy != 0 else None)
    factors["ev_to_ebitda_inv"] = (1 / ev_ebitda) if (ev_ebitda and ev_ebitda > 0) else None

    # ── Low-Volatility Safety ─────────────────────────────────────────────────
    # 252-day annualized volatility (inverted so low vol = high score)
    try:
        start_date = (date.fromisoformat(as_of) - timedelta(days=380)).isoformat()
        price_df = get_prices(conn, stock_id, start=start_date, end=as_of)
        if price_df is not None and len(price_df) >= 50:
            prices = price_df["adj_close"].dropna()
            log_returns = np.log(prices / prices.shift(1)).dropna()
            # Use up to 252 trading days of returns
            log_returns_252 = log_returns.iloc[-252:]
            if len(log_returns_252) >= 50:
                vol_252d = float(log_returns_252.std() * np.sqrt(252))
                factors["low_vol_252d_inv"] = (1.0 / vol_252d) if vol_252d > 0 else None
            else:
                factors["low_vol_252d_inv"] = None
            # Store raw vol for risk overlay (not persisted as a scored factor)
            factors["_vol_252d_raw"] = vol_252d if len(log_returns_252) >= 50 else None
        else:
            factors["low_vol_252d_inv"] = None
            factors["_vol_252d_raw"] = None
    except Exception:
        factors["low_vol_252d_inv"] = None
        factors["_vol_252d_raw"] = None

    # Debt to Equity (inverted — lower leverage = higher score)
    de = safe_div(td, eq)
    factors["debt_to_equity_inv"] = (1 / de) if (de and de > 0) else (100.0 if de == 0 else None)

    # Current Ratio = Current Assets / Current Liabilities
    factors["current_ratio"] = safe_div(ca, cl)

    # ── Income / Dividend ─────────────────────────────────────────────────────
    # Dividend Yield = abs(dividends_paid) / shares_outstanding / price
    # dividends_paid is typically a negative cash outflow in EDGAR
    try:
        price_now = None
        if market_cap and shares and shares > 0:
            price_now = market_cap / shares
        if price_now and price_now > 0 and div is not None and div != 0:
            ann_div_per_share = abs(div) / shares if shares and shares > 0 else None
            factors["dividend_yield"] = safe_div(ann_div_per_share, price_now)
        else:
            factors["dividend_yield"] = None
    except Exception:
        factors["dividend_yield"] = None

    # Dividend CAGR 5 years (20 quarters)
    try:
        if len(fdf) >= 20 and prev_20 is not None:
            div_now = fdf["dividends_paid"].iloc[:4].apply(lambda x: abs(x) if x else 0).sum()
            div_5y  = fdf["dividends_paid"].iloc[-4:].apply(lambda x: abs(x) if x else 0).sum()
            if div_5y > 0 and div_now > 0:
                factors["div_growth_5y"] = compute_cagr([div_5y, div_now], years=5)
            else:
                factors["div_growth_5y"] = None
        else:
            factors["div_growth_5y"] = None
    except Exception:
        factors["div_growth_5y"] = None

    # Consecutive dividend increases (quarters where abs(div) >= prior quarter)
    # Signals dividend aristocrat-like behavior.  Cap at 40.  0.0 if never paid.
    try:
        div_series = fdf["dividends_paid"].apply(
            lambda x: abs(x) if (x is not None and not (isinstance(x, float) and np.isnan(x))) else 0.0
        )
        # div_series is sorted newest-first; reverse to scan chronologically
        div_list = list(div_series)[::-1]

        if all(v == 0 for v in div_list):
            factors["consecutive_increases"] = 0.0
        else:
            # Walk from the newest quarter backwards and count the streak
            streak = 0
            # Flip back to newest-first so we count the current streak
            for i in range(len(div_series) - 1):
                curr_div = div_series.iloc[i]
                prev_div = div_series.iloc[i + 1]
                if curr_div >= prev_div and curr_div > 0:
                    streak += 1
                else:
                    break
            factors["consecutive_increases"] = float(min(streak, 40))
    except Exception:
        factors["consecutive_increases"] = 0.0

    return factors


def score_universe(as_of: Optional[str] = None, persist: bool = True) -> pd.DataFrame:
    """
    Score all active stocks on the conservative strategy.
    Returns DataFrame with columns: ticker, composite_score, [factor columns].
    Persists scores and predictions to DB if persist=True.
    """
    if as_of is None:
        as_of = date.today().isoformat()

    conn = get_connection()
    universe = get_active_universe(conn)
    weights = get_active_weights(conn, STRATEGY)
    weight_version = get_latest_weight_version(conn, STRATEGY) or INITIAL_WEIGHT_VERSION

    log.info(
        "Scoring %d stocks on '%s' strategy (as of %s)...",
        len(universe), STRATEGY, as_of,
    )

    # ── Step 1: Collect raw factors for all stocks ────────────────────────────
    all_factors: list[dict] = []
    for _, row in universe.iterrows():
        tkr  = row["ticker"]
        sid  = row["id"]
        mcap = row.get("market_cap")
        try:
            raw = compute_raw_factors(tkr, sid, conn, as_of, market_cap=mcap)
            raw["ticker"]     = tkr
            raw["stock_id"]   = sid
            raw["sector"]     = row.get("sector")
            raw["_market_cap"] = mcap   # Stored for risk overlay; not a scored factor
            all_factors.append(raw)
        except Exception as e:
            log.debug("Raw factor error for %s: %s", tkr, e)
            all_factors.append({"ticker": tkr, "stock_id": sid, "sector": row.get("sector"), "_market_cap": mcap})

    df = pd.DataFrame(all_factors)
    if df.empty:
        conn.close()
        return df

    # ── Step 2: Cross-sectional percentile rank each factor ──────────────────
    factor_cols = [f for f in weights if f in df.columns]
    for col in factor_cols:
        df[col + "_pct"] = percentile_rank(df[col])

    # Optional: sector-neutralize raw factor values before final percentile ranking
    if SECTOR_NEUTRALIZE and "sector" in df.columns:
        df = sector_neutralize(df, factor_cols=factor_cols, sector_col="sector")
        # Re-rank after neutralization
        for col in factor_cols:
            df[col + "_pct"] = percentile_rank(df[col])

    # ── Step 3: Compute weighted composite score ──────────────────────────────
    def _composite(row):
        factor_scores = {f: row.get(f + "_pct", _FILL) for f in weights}
        return weighted_composite_score(factor_scores, weights)

    df["composite_score"] = df.apply(_composite, axis=1)

    # ── Step 4: Risk overlay — mark exclusions ────────────────────────────────
    def _passes_risk(row):
        # Altman Z distress filter
        z = row.get("altman_z_norm")
        if z is not None and z < ALTMAN_DISTRESS:
            return False, "Altman Z < 1.81"

        # High-volatility exclusion (>45% annualized)
        vol = row.get("_vol_252d_raw")
        if vol is not None and vol > _MAX_VOL_252D:
            return False, "High volatility (>45% annual)"

        # Small-cap exclusion (<$5B market cap)
        mcap = row.get("_market_cap")
        if mcap is not None and mcap < _MIN_MARKET_CAP:
            return False, "Market cap below $5B"

        return True, ""

    df["passes_risk"], df["exclusion_reason"] = zip(*df.apply(_passes_risk, axis=1))

    # ── Step 5: Rank ──────────────────────────────────────────────────────────
    df_ranked   = df[df["passes_risk"]].copy()
    df_ranked   = df_ranked.sort_values("composite_score", ascending=False)
    df_ranked["rank"] = range(1, len(df_ranked) + 1)
    df_excluded = df[~df["passes_risk"]].copy()
    df_excluded["rank"] = None
    df_out = pd.concat([df_ranked, df_excluded], ignore_index=True)

    # ── Step 6: Persist ───────────────────────────────────────────────────────
    if persist:
        with conn:
            for _, row in df_out.iterrows():
                sid   = int(row["stock_id"])
                score = float(row["composite_score"]) if pd.notna(row.get("composite_score")) else 0.0
                comps = {f + "_pct": float(row.get(f + "_pct", _FILL)) for f in factor_cols}
                upsert_score(conn, sid, as_of, STRATEGY, score, comps, weight_version)
                rank = int(row["rank"]) if pd.notna(row.get("rank")) else 9999
                upsert_prediction(conn, sid, STRATEGY, as_of, score, rank, comps)
            conn.commit()

    conn.close()
    log.info(
        "Conservative scoring complete. Top score: %.1f (%s)",
        df_ranked["composite_score"].max() if not df_ranked.empty else 0,
        df_ranked.iloc[0]["ticker"] if not df_ranked.empty else "N/A",
    )
    return df_out
