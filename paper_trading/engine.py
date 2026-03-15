"""
Horizon Ledger — Paper Portfolio Simulation Engine
Simulates portfolio performance by mirroring the strategy indexes.

Key design decisions:
  - Backsimulated period (before live_start_date): executes at closing price, no look-ahead
    beyond what the point-in-time scoring system already enforces.
  - Live period (from live_start_date): executes at next-day OPEN price.
  - Fractional shares supported (REAL type in DB).
  - Slippage applied to all trades (configurable: PAPER_SLIPPAGE in config.py).
  - Backsimulated and live periods are always tracked and reported separately.
  - If LIVE_START_DATE is None in config, today is used (no backsim).

SIMULATION ONLY. No brokerage integration, no real orders, no financial advice.
"""

import json
import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    PAPER_SLIPPAGE,
    PAPER_STARTING_CASH,
    LIVE_START_DATE,
    BENCHMARK_TICKER,
)
from db.schema import get_connection
from db.queries import (
    create_paper_portfolio,
    get_paper_portfolio,
    get_all_paper_portfolios,
    update_paper_cash,
    upsert_paper_position,
    close_paper_position,
    get_open_paper_positions,
    log_paper_transaction,
    get_paper_transactions,
    get_current_holdings,
    get_stock_id,
    get_price_on_date,
    get_prices,
)

log = logging.getLogger(__name__)

STRATEGIES = ["long_term", "dividend", "turnaround", "swing"]

PORTFOLIO_ID_MAP = {
    "long_term":  "long_term_1000",
    "dividend":   "dividend_1000",
    "turnaround": "turnaround_1000",
    "swing":      "swing_1000",
}


# ─── Portfolio Lifecycle ──────────────────────────────────────────────────────

def initialize_all_portfolios(
    starting_cash: float = PAPER_STARTING_CASH,
    live_start_date: Optional[str] = None,
) -> None:
    """
    Create a paper portfolio for each of the 4 strategies if it doesn't exist.
    Should be called once when the system is first set up.
    """
    if live_start_date is None:
        live_start_date = LIVE_START_DATE or date.today().isoformat()

    conn = get_connection()
    with conn:
        for strategy, portfolio_id in PORTFOLIO_ID_MAP.items():
            create_paper_portfolio(
                conn,
                portfolio_id=portfolio_id,
                strategy=strategy,
                starting_cash=starting_cash,
                display_name=f"{strategy.replace('_', ' ').title()} $1,000",
                live_start_date=live_start_date,
            )
        conn.commit()
    conn.close()
    log.info("Paper portfolios initialized for %d strategies", len(STRATEGIES))


# ─── Price Lookup ─────────────────────────────────────────────────────────────

def _get_close_price(conn, stock_id: int, as_of: str) -> Optional[float]:
    """Get adjusted close price on or before as_of."""
    return get_price_on_date(conn, stock_id, as_of)


def _get_open_price(conn, stock_id: int, target_date: str) -> Optional[float]:
    """
    Get the opening price on target_date (next-day open for live trades).
    Falls back to close price if open is not available.
    """
    row = conn.execute(
        """SELECT open, adj_close FROM daily_prices
           WHERE stock_id=? AND date=? LIMIT 1""",
        (stock_id, target_date),
    ).fetchone()
    if row:
        return row["open"] or row["adj_close"]
    # If exact date unavailable, use close on or before
    return get_price_on_date(conn, stock_id, target_date)


# ─── Portfolio Value ──────────────────────────────────────────────────────────

def get_portfolio_value(portfolio_id: str, as_of: Optional[str] = None) -> float:
    """
    Compute current total portfolio value: sum of (shares × close price) + cash.
    Returns 0.0 if portfolio doesn't exist.
    """
    if as_of is None:
        as_of = date.today().isoformat()

    conn = get_connection()
    portfolio = get_paper_portfolio(conn, portfolio_id)
    if not portfolio:
        conn.close()
        return 0.0

    positions = get_open_paper_positions(conn, portfolio_id)
    holdings_value = 0.0
    for _, pos in positions.iterrows():
        price = _get_close_price(conn, int(pos["stock_id"]), as_of)
        if price:
            holdings_value += float(pos["shares"]) * price

    cash = float(portfolio["current_cash"] or 0)
    conn.close()
    return holdings_value + cash


def get_equity_curve(
    portfolio_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute daily portfolio value from transactions and positions.
    Also returns SPY equity curve starting at the same dollar value.

    Returns DataFrame with columns: date, portfolio_value, spy_value,
    is_backsimulated (1=before live_start, 0=after).
    """
    conn = get_connection()
    portfolio = get_paper_portfolio(conn, portfolio_id)
    if not portfolio:
        conn.close()
        return pd.DataFrame()

    live_start = portfolio["live_start_date"] or date.today().isoformat()
    if start_date is None:
        # Start from first transaction or 5 years ago
        first_tx = conn.execute(
            "SELECT MIN(date) FROM paper_transactions WHERE portfolio_id=?",
            (portfolio_id,),
        ).fetchone()[0]
        start_date = first_tx or (date.today() - timedelta(days=365)).isoformat()

    if end_date is None:
        end_date = date.today().isoformat()

    # Build date range
    dates = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days
    if len(dates) == 0:
        conn.close()
        return pd.DataFrame()

    # Reconstruct daily portfolio value using transaction history
    # Simpler approach: replay transactions chronologically
    transactions = get_paper_transactions(conn, portfolio_id,
                                          start=start_date, end=end_date)

    # Get all open positions and their full price history
    all_positions_df = pd.read_sql(
        """SELECT pp.*, s.ticker FROM paper_positions pp
           JOIN stocks s ON s.id = pp.stock_id
           WHERE pp.portfolio_id=?""",
        conn, params=[portfolio_id],
    )

    curve_rows = []
    starting_cash = float(portfolio["starting_cash"])

    for d in dates:
        d_str = d.strftime("%Y-%m-%d")
        val = _reconstruct_value_on_date(
            conn, portfolio_id, portfolio, all_positions_df, d_str
        )
        curve_rows.append({
            "date":               d_str,
            "portfolio_value":    val,
            "is_backsimulated":   1 if d_str < live_start else 0,
        })

    # SPY benchmark — same starting value
    spy_id = get_stock_id(conn, BENCHMARK_TICKER)
    spy_start_price = None
    if spy_id:
        spy_start_price = _get_close_price(conn, spy_id, start_date)

    conn.close()

    result = pd.DataFrame(curve_rows)
    if result.empty:
        return result

    # Normalize portfolio to starting_cash at first valid point
    first_valid = result["portfolio_value"].replace(0, np.nan).first_valid_index()
    if first_valid is not None and result.loc[first_valid, "portfolio_value"] > 0:
        result["portfolio_value_norm"] = (
            result["portfolio_value"] / result.loc[first_valid, "portfolio_value"]
            * starting_cash
        )
    else:
        result["portfolio_value_norm"] = result["portfolio_value"]

    # SPY curve
    if spy_id and spy_start_price:
        conn2 = get_connection()
        spy_prices = get_prices(conn2, spy_id, start=start_date, end=end_date)
        conn2.close()
        if not spy_prices.empty:
            spy_prices = spy_prices.set_index("date")["adj_close"]
            spy_base = spy_prices.iloc[0] if not spy_prices.empty else None
            spy_curve = spy_prices / spy_base * starting_cash if spy_base else pd.Series()
            result["spy_value"] = result["date"].map(spy_curve)
    else:
        result["spy_value"] = np.nan

    return result


def _reconstruct_value_on_date(
    conn, portfolio_id: str, portfolio: dict,
    all_positions: pd.DataFrame, d_str: str
) -> float:
    """
    Approximate portfolio value on a given date by computing:
    positions open on that date × price + cash accumulated to that date.
    """
    try:
        starting_cash = float(portfolio["starting_cash"])

        # Cash = starting + sum of all cash_impacts up to and including d_str
        cash_row = conn.execute(
            """SELECT COALESCE(SUM(cash_impact), 0) as total_impact
               FROM paper_transactions
               WHERE portfolio_id=? AND date<=?""",
            (portfolio_id, d_str),
        ).fetchone()
        cash = starting_cash + float(cash_row["total_impact"] or 0)

        # Positions open on d_str (entered before or on d_str, not yet exited)
        pos_on_date = all_positions[
            (all_positions["entry_date"] <= d_str) & (all_positions["is_open"] == 1)
        ]
        # Also include positions that were closed after d_str
        closed_after = all_positions[
            (all_positions["entry_date"] <= d_str) & (all_positions["is_open"] == 0)
        ]
        # Closed positions: need to check close date... approximate with entry only
        # (full reconstruction would require close dates stored in positions)
        active = pos_on_date

        holdings_value = 0.0
        for _, pos in active.iterrows():
            price = _get_close_price(conn, int(pos["stock_id"]), d_str)
            if price and float(pos["shares"] or 0) > 0:
                holdings_value += float(pos["shares"]) * price

        return max(0.0, cash + holdings_value)
    except Exception:
        return 0.0


# ─── Allocation & Rebalancing ─────────────────────────────────────────────────

def allocate(
    portfolio_id: str,
    as_of: str,
    target_weights: dict,
    is_backsimulated: bool = False,
) -> dict:
    """
    Allocate the portfolio to target_weights on as_of date.
    target_weights: {ticker: weight_float} — must sum to ~1.0.

    For backsimulated: uses closing price on as_of.
    For live: uses opening price of the NEXT business day.

    Returns summary dict with trades executed.
    """
    conn = get_connection()
    portfolio = get_paper_portfolio(conn, portfolio_id)
    if not portfolio:
        conn.close()
        return {"error": f"Portfolio {portfolio_id} not found"}

    portfolio_value = get_portfolio_value(portfolio_id, as_of=as_of)
    if portfolio_value <= 0:
        portfolio_value = float(portfolio["starting_cash"])

    live_start = portfolio["live_start_date"] or date.today().isoformat()
    is_backsim  = is_backsimulated or as_of < live_start

    # Execution date: same day for backsim, next business day for live
    if is_backsim:
        exec_date = as_of
    else:
        exec_date = _next_business_day(as_of)

    trades = []
    with conn:
        for ticker, weight in target_weights.items():
            sid = get_stock_id(conn, ticker)
            if sid is None:
                log.warning("Ticker %s not in DB, skipping", ticker)
                continue

            # Get execution price
            if is_backsim:
                ref_price = _get_close_price(conn, sid, exec_date)
            else:
                ref_price = _get_open_price(conn, sid, exec_date)

            if not ref_price or ref_price <= 0:
                log.warning("No price for %s on %s, skipping", ticker, exec_date)
                continue

            # Apply slippage
            exec_price = ref_price * (1 + PAPER_SLIPPAGE)

            dollar_alloc = portfolio_value * weight
            shares = dollar_alloc / exec_price
            cash_impact = -(shares * exec_price)

            upsert_paper_position(conn, portfolio_id, sid, shares, exec_price, exec_date)
            log_paper_transaction(
                conn, portfolio_id, exec_date, "BUY", sid,
                shares, ref_price, exec_price, cash_impact,
                "initial_allocation", int(is_backsim),
            )
            trades.append({"ticker": ticker, "shares": shares,
                           "price": exec_price, "value": dollar_alloc})

        # Update cash: starting_cash - sum of purchases
        total_spent = sum(abs(t["value"]) for t in trades) * (1 + PAPER_SLIPPAGE)
        new_cash = float(portfolio["current_cash"] or portfolio["starting_cash"]) - total_spent
        update_paper_cash(conn, portfolio_id, max(0.0, new_cash))
        conn.commit()

    conn.close()
    log.info("Allocated %s: %d positions on %s (backsim=%s)",
             portfolio_id, len(trades), exec_date, is_backsim)
    return {"trades": trades, "exec_date": exec_date, "backsim": is_backsim}


def rebalance(
    portfolio_id: str,
    as_of: str,
    new_target_weights: dict,
    reason: str = "quarterly_recon",
) -> dict:
    """
    Rebalance portfolio to new target weights.
    Sells positions no longer needed, buys new/increased positions.
    Applies PAPER_SLIPPAGE to each trade.
    """
    conn = get_connection()
    portfolio = get_paper_portfolio(conn, portfolio_id)
    if not portfolio:
        conn.close()
        return {}

    live_start = portfolio["live_start_date"] or date.today().isoformat()
    is_backsim  = as_of < live_start
    exec_date   = as_of if is_backsim else _next_business_day(as_of)

    current_positions = get_open_paper_positions(conn, portfolio_id)
    portfolio_value   = get_portfolio_value(portfolio_id, as_of=as_of)

    sells = []
    buys  = []

    with conn:
        # ── Sell positions not in new target ─────────────────────────────────
        for _, pos in current_positions.iterrows():
            ticker = pos["ticker"]
            sid    = int(pos["stock_id"])
            shares = float(pos["shares"])

            if ticker not in new_target_weights:
                # Full exit
                ref_price  = _get_close_price(conn, sid, exec_date) if is_backsim \
                             else _get_open_price(conn, sid, exec_date)
                if not ref_price:
                    continue
                exec_price = ref_price * (1 - PAPER_SLIPPAGE)
                cash_in    = shares * exec_price

                close_paper_position(conn, portfolio_id, sid)
                log_paper_transaction(
                    conn, portfolio_id, exec_date, "SELL", sid,
                    shares, ref_price, exec_price, cash_in, reason, int(is_backsim),
                )
                sells.append({"ticker": ticker, "shares": shares, "cash_in": cash_in})

        # ── Buy / adjust remaining positions ─────────────────────────────────
        new_cash = float(portfolio["current_cash"] or 0)
        new_cash += sum(s["cash_in"] for s in sells)

        for ticker, weight in new_target_weights.items():
            sid = get_stock_id(conn, ticker)
            if sid is None:
                continue

            ref_price  = _get_close_price(conn, sid, exec_date) if is_backsim \
                         else _get_open_price(conn, sid, exec_date)
            if not ref_price:
                continue
            exec_price = ref_price * (1 + PAPER_SLIPPAGE)

            dollar_alloc = portfolio_value * weight
            new_shares   = dollar_alloc / exec_price
            cash_out     = -(new_shares * exec_price)

            upsert_paper_position(conn, portfolio_id, sid, new_shares, exec_price, exec_date)
            log_paper_transaction(
                conn, portfolio_id, exec_date, "REBALANCE", sid,
                new_shares, ref_price, exec_price, cash_out, reason, int(is_backsim),
            )
            buys.append({"ticker": ticker, "shares": new_shares})
            new_cash += cash_out

        update_paper_cash(conn, portfolio_id, max(0.0, new_cash))
        conn.commit()

    conn.close()
    log.info("Rebalanced %s: %d sells, %d buys on %s",
             portfolio_id, len(sells), len(buys), exec_date)
    return {"sells": sells, "buys": buys, "exec_date": exec_date}


# ─── Performance Metrics ──────────────────────────────────────────────────────

def get_performance_metrics(
    portfolio_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """
    Compute performance metrics for the portfolio.
    Returns separate metrics for backsimulated period, live period, and combined.
    Metrics: total_return, annualized_return, volatility, sharpe, sortino,
             max_drawdown, beta_to_spy, alpha, information_ratio.
    """
    if end_date is None:
        end_date = date.today().isoformat()

    conn = get_connection()
    portfolio = get_paper_portfolio(conn, portfolio_id)
    conn.close()
    if not portfolio:
        return {}

    live_start = portfolio["live_start_date"] or date.today().isoformat()

    curve = get_equity_curve(portfolio_id, start_date=start_date, end_date=end_date)
    if curve.empty:
        return {}

    def _compute_metrics(sub_curve: pd.DataFrame, label: str) -> dict:
        if len(sub_curve) < 5:
            return {label: {"insufficient_data": True}}
        vals = sub_curve["portfolio_value"].dropna()
        if vals.iloc[0] <= 0:
            return {label: {"insufficient_data": True}}

        returns = vals.pct_change().dropna()
        total_return  = (vals.iloc[-1] / vals.iloc[0]) - 1
        n_days        = len(returns)
        ann_return    = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        volatility    = returns.std() * np.sqrt(252)
        downside_ret  = returns[returns < 0]
        downside_vol  = downside_ret.std() * np.sqrt(252) if len(downside_ret) > 5 else volatility

        # Risk-free rate (approximate 4% annual → daily)
        rf_daily = 0.04 / 252
        sharpe   = (returns.mean() - rf_daily) / returns.std() * np.sqrt(252) \
                   if returns.std() > 0 else 0
        sortino  = (returns.mean() - rf_daily) / (downside_vol / np.sqrt(252)) \
                   if downside_vol > 0 else 0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        roll_max   = cumulative.cummax()
        drawdowns  = (cumulative - roll_max) / roll_max
        max_dd     = float(drawdowns.min())

        # Beta / alpha vs SPY
        spy_vals = sub_curve["spy_value"].dropna()
        beta = alpha = np.nan
        if len(spy_vals) > 10:
            spy_rets = spy_vals.pct_change().dropna()
            aligned  = returns.align(spy_rets, join="inner")
            if len(aligned[0]) > 10:
                cov   = np.cov(aligned[0], aligned[1])
                beta  = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan
                alpha = (returns.mean() - rf_daily - beta * (spy_rets.mean() - rf_daily)) * 252

        return {
            label: {
                "total_return":    round(total_return, 4),
                "annualized_return": round(ann_return, 4),
                "volatility":      round(volatility, 4),
                "sharpe":          round(sharpe, 3),
                "sortino":         round(sortino, 3),
                "max_drawdown":    round(max_dd, 4),
                "beta":            round(beta, 3) if not np.isnan(beta) else None,
                "alpha":           round(alpha, 4) if not np.isnan(alpha) else None,
                "n_trading_days":  n_days,
            }
        }

    metrics = {}

    # Combined
    metrics.update(_compute_metrics(curve, "combined"))

    # Backsimulated period
    backsim_curve = curve[curve["is_backsimulated"] == 1]
    if len(backsim_curve) >= 5:
        metrics.update(_compute_metrics(backsim_curve.reset_index(drop=True), "backsimulated"))

    # Live period
    live_curve = curve[curve["is_backsimulated"] == 0]
    if len(live_curve) >= 5:
        metrics.update(_compute_metrics(live_curve.reset_index(drop=True), "live"))

    return metrics


# ─── Mirror Index Reconstitution ─────────────────────────────────────────────

def sync_portfolio_to_index(portfolio_id: str, strategy: str, as_of: Optional[str] = None) -> dict:
    """
    Mirror current index holdings to the paper portfolio.
    Called after each quarterly reconstitution.
    """
    if as_of is None:
        as_of = date.today().isoformat()

    conn = get_connection()
    holdings = get_current_holdings(conn, f"{strategy}_index")
    conn.close()

    if holdings.empty:
        return {"message": "No index holdings to mirror"}

    # Build target weights from current holdings
    total_weight = holdings["target_weight"].sum()
    if total_weight <= 0:
        return {"message": "Zero total weight in index"}

    target_weights = {
        row["ticker"]: float(row["target_weight"]) / total_weight
        for _, row in holdings.iterrows()
        if row["target_weight"] and row["target_weight"] > 0
    }

    return rebalance(portfolio_id, as_of, target_weights, reason="index_sync")


def sync_all_portfolios(as_of: Optional[str] = None) -> None:
    """Sync all four paper portfolios to their corresponding indexes."""
    if as_of is None:
        as_of = date.today().isoformat()

    initialize_all_portfolios()  # No-op if already created

    for strategy, portfolio_id in PORTFOLIO_ID_MAP.items():
        try:
            result = sync_portfolio_to_index(portfolio_id, strategy, as_of)
            log.info("Synced %s: %s", portfolio_id, result)
        except Exception as e:
            log.error("Failed to sync %s: %s", portfolio_id, e)


# ─── Utilities ────────────────────────────────────────────────────────────────

def _next_business_day(d: str) -> str:
    """Return the next business day after d (Mon-Fri, no holiday check)."""
    dt = date.fromisoformat(d) + timedelta(days=1)
    while dt.weekday() >= 5:  # 5=Saturday, 6=Sunday
        dt += timedelta(days=1)
    return dt.isoformat()
