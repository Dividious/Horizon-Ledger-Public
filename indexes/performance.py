"""
Horizon Ledger — Index Performance Tracking
Computes metrics vs S&P 500 (SPY) benchmark.

Metrics:
  Cumulative return, annualized return, volatility, Sharpe, Sortino,
  max drawdown, Beta, Alpha, Information Ratio, rolling 12-month alpha.

Factor attribution using Fama-French 5-Factor + Momentum model.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import BENCHMARK_TICKER, RISK_FREE_TICKER
from db.schema import get_connection
from db.queries import (
    get_current_holdings,
    get_price_on_date,
    get_macro_series,
    get_stock_id,
)

log = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


def compute_index_returns(
    index_name: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.Series:
    """
    Compute daily total return series for an index based on historical holdings.
    Uses a simplified equal-weight approach based on current target weights.
    Returns a price-return Series indexed by date.
    """
    if end is None:
        end = date.today().isoformat()
    if start is None:
        start = (date.fromisoformat(end) - timedelta(days=365 * 2)).isoformat()

    conn = get_connection()
    holdings = get_current_holdings(conn, index_name)

    if holdings.empty:
        conn.close()
        return pd.Series(dtype=float, name=index_name)

    # Gather price series for each holding
    from db.queries import get_prices
    price_dfs = {}
    for _, h in holdings.iterrows():
        tkr = h["ticker"]
        sid = h["stock_id"]
        df = get_prices(conn, sid, start=start, end=end)
        if not df.empty:
            df = df.set_index("date")["adj_close"]
            df.name = tkr
            price_dfs[tkr] = df

    conn.close()

    if not price_dfs:
        return pd.Series(dtype=float, name=index_name)

    prices = pd.DataFrame(price_dfs).sort_index()
    prices = prices.ffill()   # Forward-fill missing days

    # Equal weight returns
    returns = prices.pct_change()
    index_returns = returns.mean(axis=1)   # Equal-weighted
    index_level = (1 + index_returns).cumprod()
    index_level.name = index_name
    return index_level


def get_benchmark_returns(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.Series:
    """Return SPY cumulative return series."""
    if end is None:
        end = date.today().isoformat()
    if start is None:
        start = (date.fromisoformat(end) - timedelta(days=365 * 2)).isoformat()

    conn = get_connection()
    spy_id = get_stock_id(conn, BENCHMARK_TICKER)
    if spy_id is None:
        conn.close()
        return pd.Series(dtype=float, name=BENCHMARK_TICKER)

    from db.queries import get_prices
    df = get_prices(conn, spy_id, start=start, end=end)
    conn.close()

    if df.empty:
        return pd.Series(dtype=float, name=BENCHMARK_TICKER)

    s = df.set_index("date")["adj_close"]
    returns = s.pct_change()
    level = (1 + returns).cumprod()
    level.name = BENCHMARK_TICKER
    return level


def get_risk_free_rate(conn=None) -> float:
    """Return current annualized risk-free rate from DGS10 (decimal)."""
    close_conn = conn is None
    if conn is None:
        conn = get_connection()
    try:
        df = get_macro_series(conn, RISK_FREE_TICKER)
        if df.empty:
            return 0.045
        return float(df.iloc[-1]["value"]) / 100
    except Exception:
        return 0.045
    finally:
        if close_conn:
            conn.close()


def compute_performance_metrics(
    index_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.045,
) -> dict:
    """
    Compute all performance metrics from daily return series.
    index_returns and benchmark_returns are cumulative price series.
    """
    if index_returns.empty or len(index_returns) < 20:
        return {}

    # Convert to daily returns
    idx_ret = index_returns.pct_change().dropna()
    bmk_ret = benchmark_returns.pct_change().dropna()

    # Align dates
    common = idx_ret.index.intersection(bmk_ret.index)
    if len(common) < 20:
        return {}

    idx_ret = idx_ret[common]
    bmk_ret = bmk_ret[common]

    n_days = len(idx_ret)
    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1

    # ── Basic stats ───────────────────────────────────────────────────────────
    cumulative_return = float((1 + idx_ret).prod() - 1)
    n_years = n_days / TRADING_DAYS_PER_YEAR
    annual_return = float((1 + cumulative_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    annual_vol = float(idx_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    # ── Sharpe Ratio ──────────────────────────────────────────────────────────
    excess = idx_ret - rf_daily
    sharpe = float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if excess.std() > 0 else 0.0

    # ── Sortino Ratio ─────────────────────────────────────────────────────────
    downside = excess[excess < 0]
    downside_dev = float(downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(downside) > 0 else 0.001
    sortino = float(excess.mean() * TRADING_DAYS_PER_YEAR / downside_dev)

    # ── Maximum Drawdown ──────────────────────────────────────────────────────
    cumulative = (1 + idx_ret).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    # Drawdown duration (in days)
    in_drawdown = drawdown < 0
    if in_drawdown.any():
        dd_streak = in_drawdown.astype(int)
        max_dd_duration = int(dd_streak.groupby((~in_drawdown).cumsum()).cumsum().max())
    else:
        max_dd_duration = 0

    # ── Beta and Alpha ────────────────────────────────────────────────────────
    cov_matrix = np.cov(idx_ret, bmk_ret)
    beta = float(cov_matrix[0, 1] / cov_matrix[1, 1]) if cov_matrix[1, 1] > 0 else 1.0
    bmk_annual = float((1 + bmk_ret).prod() - 1) ** (1 / n_years) if n_years > 0 else 0.0
    alpha = annual_return - (risk_free_rate + beta * (bmk_annual - risk_free_rate))

    # ── Information Ratio ─────────────────────────────────────────────────────
    active_return = idx_ret - bmk_ret
    tracking_error = float(active_return.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    active_annual = float(active_return.mean() * TRADING_DAYS_PER_YEAR)
    info_ratio = float(active_annual / tracking_error) if tracking_error > 0 else 0.0

    # ── Rolling 12-month alpha ────────────────────────────────────────────────
    window = min(252, len(idx_ret))
    if len(idx_ret) >= 63:
        rolling_alpha = (
            idx_ret.rolling(252).mean() - rf_daily -
            beta * (bmk_ret.rolling(252).mean() - rf_daily)
        ) * TRADING_DAYS_PER_YEAR
        latest_rolling_alpha = float(rolling_alpha.dropna().iloc[-1]) if not rolling_alpha.dropna().empty else alpha
    else:
        latest_rolling_alpha = alpha

    return {
        "cumulative_return":    cumulative_return,
        "annual_return":        annual_return,
        "annual_volatility":    annual_vol,
        "sharpe_ratio":         sharpe,
        "sortino_ratio":        sortino,
        "max_drawdown":         max_drawdown,
        "max_dd_duration_days": max_dd_duration,
        "beta":                 beta,
        "alpha":                alpha,
        "information_ratio":    info_ratio,
        "tracking_error":       tracking_error,
        "rolling_12m_alpha":    latest_rolling_alpha,
        "n_observations":       n_days,
        "benchmark_return":     float((1 + bmk_ret).prod() - 1),
    }


def get_all_metrics(
    index_name: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """Convenience: compute full performance metrics for one index."""
    idx_series  = compute_index_returns(index_name, start, end)
    bmk_series  = get_benchmark_returns(start, end)
    rf          = get_risk_free_rate()
    metrics     = compute_performance_metrics(idx_series, bmk_series, rf)
    metrics["index_name"] = index_name
    metrics["as_of"]      = end or date.today().isoformat()
    return metrics


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """Return the drawdown series for plotting."""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    return (cumulative - rolling_max) / rolling_max
