"""
Horizon Ledger — Reusable Query Functions
All SQL access goes through here to keep the rest of the codebase clean.
"""

import json
import sqlite3
import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from db.schema import get_connection
from config import DB_PATH, EDGAR_FUNDAMENTALS_LAG_DAYS

log = logging.getLogger(__name__)


# ─── Stocks ───────────────────────────────────────────────────────────────────

def upsert_stock(conn: sqlite3.Connection, ticker: str, **kwargs) -> int:
    """Insert or update a stock row. Returns the stock_id."""
    cols = ["ticker"] + list(kwargs.keys())
    vals = [ticker] + list(kwargs.values())
    placeholders = ",".join(["?"] * len(vals))
    updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "ticker")
    conn.execute(
        f"""INSERT INTO stocks ({','.join(cols)}) VALUES ({placeholders})
            ON CONFLICT(ticker) DO UPDATE SET {updates}""",
        vals,
    )
    row = conn.execute("SELECT id FROM stocks WHERE ticker=?", (ticker,)).fetchone()
    return row["id"]


def get_stock_id(conn: sqlite3.Connection, ticker: str) -> Optional[int]:
    row = conn.execute("SELECT id FROM stocks WHERE ticker=?", (ticker,)).fetchone()
    return row["id"] if row else None


def get_active_universe(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return all active stocks as a DataFrame."""
    return pd.read_sql(
        "SELECT * FROM stocks WHERE is_active=1 ORDER BY ticker",
        conn,
    )


def get_universe_tickers(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT ticker FROM stocks WHERE is_active=1 ORDER BY ticker"
    ).fetchall()
    return [r["ticker"] for r in rows]


# ─── Prices ───────────────────────────────────────────────────────────────────

def upsert_prices(conn: sqlite3.Connection, stock_id: int, df: pd.DataFrame) -> int:
    """
    Bulk upsert OHLCV data.  df must have columns:
    date, open, high, low, close, adj_close, volume
    Returns number of rows inserted/replaced.
    """
    records = df.assign(stock_id=stock_id)[
        ["stock_id", "date", "open", "high", "low", "close", "adj_close", "volume"]
    ].values.tolist()
    conn.executemany(
        """INSERT INTO daily_prices
           (stock_id,date,open,high,low,close,adj_close,volume)
           VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT(stock_id,date) DO UPDATE SET
             open=excluded.open, high=excluded.high, low=excluded.low,
             close=excluded.close, adj_close=excluded.adj_close,
             volume=excluded.volume""",
        records,
    )
    return len(records)


def get_prices(
    conn: sqlite3.Connection,
    stock_id: int,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    query = "SELECT * FROM daily_prices WHERE stock_id=?"
    params: list = [stock_id]
    if start:
        query += " AND date>=?"
        params.append(start)
    if end:
        query += " AND date<=?"
        params.append(end)
    query += " ORDER BY date"
    return pd.read_sql(query, conn, params=params)


def get_latest_price(conn: sqlite3.Connection, stock_id: int) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM daily_prices WHERE stock_id=? ORDER BY date DESC LIMIT 1",
        (stock_id,),
    ).fetchone()
    return dict(row) if row else None


def get_price_on_date(conn: sqlite3.Connection, stock_id: int, as_of: str) -> Optional[float]:
    """Return adj_close on or before as_of date."""
    row = conn.execute(
        """SELECT adj_close FROM daily_prices
           WHERE stock_id=? AND date<=? ORDER BY date DESC LIMIT 1""",
        (stock_id, as_of),
    ).fetchone()
    return row["adj_close"] if row else None


# ─── Fundamentals ─────────────────────────────────────────────────────────────

def upsert_fundamental(conn: sqlite3.Connection, stock_id: int, data: dict) -> None:
    """Insert or replace a single fundamentals row."""
    data["stock_id"] = stock_id
    cols = list(data.keys())
    placeholders = ",".join(["?"] * len(cols))
    updates = ",".join(
        f"{c}=excluded.{c}" for c in cols if c not in ("stock_id", "report_date", "fiscal_quarter")
    )
    conn.execute(
        f"""INSERT INTO fundamentals ({','.join(cols)}) VALUES ({placeholders})
            ON CONFLICT(stock_id, report_date, fiscal_quarter) DO UPDATE SET {updates}""",
        list(data.values()),
    )


def get_fundamentals_as_of(
    conn: sqlite3.Connection,
    stock_id: int,
    as_of_date: str,
    n_quarters: int = 8,
    lag_days: int = EDGAR_FUNDAMENTALS_LAG_DAYS,
) -> pd.DataFrame:
    """
    Point-in-time fundamentals: only return rows where
      filing_date <= as_of_date - lag_days
    This enforces no look-ahead bias.

    Reference: Leinweber (2007) — point-in-time data discipline
    """
    cutoff = (
        date.fromisoformat(as_of_date) - timedelta(days=lag_days)
    ).isoformat()
    return pd.read_sql(
        """SELECT * FROM fundamentals
           WHERE stock_id=? AND filing_date<=?
           ORDER BY report_date DESC
           LIMIT ?""",
        conn,
        params=[stock_id, cutoff, n_quarters],
    )


def get_latest_fundamentals(
    conn: sqlite3.Connection,
    stock_id: int,
    as_of_date: Optional[str] = None,
) -> Optional[pd.Series]:
    """Return the single most recent point-in-time fundamental row."""
    if as_of_date is None:
        as_of_date = date.today().isoformat()
    df = get_fundamentals_as_of(conn, stock_id, as_of_date, n_quarters=1)
    if df.empty:
        return None
    return df.iloc[0]


# ─── Scores ───────────────────────────────────────────────────────────────────

def upsert_score(
    conn: sqlite3.Connection,
    stock_id: int,
    score_date: str,
    strategy: str,
    composite_score: float,
    score_components: dict,
    weights_version: str,
) -> None:
    conn.execute(
        """INSERT INTO scores
           (stock_id,score_date,strategy,composite_score,score_components,weights_version)
           VALUES (?,?,?,?,?,?)
           ON CONFLICT(stock_id,score_date,strategy) DO UPDATE SET
             composite_score=excluded.composite_score,
             score_components=excluded.score_components,
             weights_version=excluded.weights_version""",
        (
            stock_id, score_date, strategy,
            composite_score, json.dumps(score_components), weights_version,
        ),
    )


def get_latest_scores(conn: sqlite3.Connection, strategy: str) -> pd.DataFrame:
    """Return the most recent score for each stock for a given strategy."""
    return pd.read_sql(
        """SELECT s.ticker, s.name, s.sector, s.industry,
                  sc.score_date, sc.composite_score, sc.score_components,
                  sc.weights_version, sc.stock_id
           FROM scores sc
           JOIN stocks s ON s.id = sc.stock_id
           WHERE sc.strategy=?
             AND sc.score_date = (
               SELECT MAX(score_date) FROM scores
               WHERE strategy=? AND stock_id=sc.stock_id
             )
           ORDER BY sc.composite_score DESC""",
        conn,
        params=[strategy, strategy],
    )


# ─── Predictions ──────────────────────────────────────────────────────────────

def upsert_prediction(
    conn: sqlite3.Connection,
    stock_id: int,
    strategy: str,
    signal_date: str,
    composite_score: float,
    score_rank: int,
    score_components: dict,
) -> None:
    conn.execute(
        """INSERT INTO predictions
           (stock_id,strategy,signal_date,composite_score,score_rank,score_components)
           VALUES (?,?,?,?,?,?)
           ON CONFLICT(stock_id,strategy,signal_date) DO UPDATE SET
             composite_score=excluded.composite_score,
             score_rank=excluded.score_rank,
             score_components=excluded.score_components""",
        (
            stock_id, strategy, signal_date,
            composite_score, score_rank, json.dumps(score_components),
        ),
    )


def get_unfilled_predictions(
    conn: sqlite3.Connection,
    horizon_field: str,
    min_days_ago: int,
    as_of: Optional[str] = None,
) -> pd.DataFrame:
    """Return predictions that are old enough but haven't had their return filled."""
    if as_of is None:
        as_of = date.today().isoformat()
    cutoff = (
        date.fromisoformat(as_of) - timedelta(days=min_days_ago)
    ).isoformat()
    return pd.read_sql(
        f"""SELECT p.*, s.ticker FROM predictions p
            JOIN stocks s ON s.id = p.stock_id
            WHERE p.{horizon_field} IS NULL
              AND p.signal_date <= ?""",
        conn,
        params=[cutoff],
    )


def fill_prediction_return(
    conn: sqlite3.Connection,
    prediction_id: int,
    horizon_field: str,
    return_value: float,
) -> None:
    conn.execute(
        f"""UPDATE predictions SET {horizon_field}=?, filled_date=?
            WHERE id=?""",
        (return_value, date.today().isoformat(), prediction_id),
    )


def get_predictions_with_returns(
    conn: sqlite3.Connection,
    strategy: str,
    horizon: str = "return_63d",
    min_rows: int = 0,
) -> pd.DataFrame:
    """Return all predictions for a strategy that have the specified return filled."""
    df = pd.read_sql(
        f"""SELECT p.*, s.ticker, s.sector FROM predictions p
            JOIN stocks s ON s.id = p.stock_id
            WHERE p.strategy=? AND p.{horizon} IS NOT NULL""",
        conn,
        params=[strategy],
    )
    if len(df) < min_rows:
        log.warning(
            "Only %d predictions with %s filled for strategy '%s' (need %d)",
            len(df), horizon, strategy, min_rows,
        )
    return df


# ─── Index Holdings ───────────────────────────────────────────────────────────

def get_current_holdings(conn: sqlite3.Connection, index_name: str) -> pd.DataFrame:
    return pd.read_sql(
        """SELECT ih.*, s.ticker, s.name, s.sector
           FROM index_holdings ih
           JOIN stocks s ON s.id = ih.stock_id
           WHERE ih.index_name=? AND ih.exit_date IS NULL
           ORDER BY ih.target_weight DESC""",
        conn,
        params=[index_name],
    )


def add_holding(
    conn: sqlite3.Connection,
    index_name: str,
    stock_id: int,
    entry_date: str,
    target_weight: float,
    entry_price: Optional[float],
    entry_score: Optional[float],
) -> None:
    conn.execute(
        """INSERT INTO index_holdings
           (index_name,stock_id,entry_date,target_weight,entry_price,entry_score)
           VALUES (?,?,?,?,?,?)""",
        (index_name, stock_id, entry_date, target_weight, entry_price, entry_score),
    )


def close_holding(
    conn: sqlite3.Connection,
    holding_id: int,
    exit_date: str,
    exit_price: Optional[float],
) -> None:
    conn.execute(
        "UPDATE index_holdings SET exit_date=?, exit_price=? WHERE id=?",
        (exit_date, exit_price, holding_id),
    )


# ─── Weight Versions ─────────────────────────────────────────────────────────

def get_active_weights(conn: sqlite3.Connection, strategy: str) -> dict:
    """Return the latest approved factor weights for a strategy."""
    row = conn.execute(
        """SELECT weights FROM weight_versions
           WHERE strategy=?
           ORDER BY created_date DESC LIMIT 1""",
        (strategy,),
    ).fetchone()
    if row:
        return json.loads(row["weights"])
    from config import STRATEGY_WEIGHTS
    return STRATEGY_WEIGHTS[strategy]


def get_latest_weight_version(conn: sqlite3.Connection, strategy: str) -> Optional[str]:
    row = conn.execute(
        """SELECT version_id FROM weight_versions
           WHERE strategy=?
           ORDER BY created_date DESC LIMIT 1""",
        (strategy,),
    ).fetchone()
    return row["version_id"] if row else None


# ─── Macro Data ───────────────────────────────────────────────────────────────

def upsert_macro(conn: sqlite3.Connection, series: str, date_str: str, value: float) -> None:
    conn.execute(
        """INSERT INTO macro_data (series,date,value) VALUES (?,?,?)
           ON CONFLICT(series,date) DO UPDATE SET value=excluded.value""",
        (series, date_str, value),
    )


def get_macro_series(
    conn: sqlite3.Connection,
    series: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    query = "SELECT date, value FROM macro_data WHERE series=?"
    params: list = [series]
    if start:
        query += " AND date>=?"
        params.append(start)
    if end:
        query += " AND date<=?"
        params.append(end)
    query += " ORDER BY date"
    return pd.read_sql(query, conn, params=params)


# ─── Regime ───────────────────────────────────────────────────────────────────

def upsert_regime(
    conn: sqlite3.Connection,
    date_str: str,
    regime: str,
    prob_bear: float,
    prob_neutral: float,
    prob_bull: float,
) -> None:
    conn.execute(
        """INSERT INTO regime_history (date,regime,prob_bear,prob_neutral,prob_bull)
           VALUES (?,?,?,?,?)
           ON CONFLICT(date) DO UPDATE SET
             regime=excluded.regime,
             prob_bear=excluded.prob_bear,
             prob_neutral=excluded.prob_neutral,
             prob_bull=excluded.prob_bull""",
        (date_str, regime, prob_bear, prob_neutral, prob_bull),
    )


def get_current_regime(conn: sqlite3.Connection) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM regime_history ORDER BY date DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


# ─── Market Digest ────────────────────────────────────────────────────────────

def upsert_digest(conn: sqlite3.Connection, data: dict) -> None:
    """Insert or replace a market digest snapshot row."""
    cols = list(data.keys())
    placeholders = ",".join(["?"] * len(cols))
    updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "date")
    conn.execute(
        f"""INSERT INTO market_digest_history ({','.join(cols)}) VALUES ({placeholders})
            ON CONFLICT(date) DO UPDATE SET {updates}""",
        list(data.values()),
    )


def get_latest_digest(conn: sqlite3.Connection) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM market_digest_history ORDER BY date DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def get_digest_history(
    conn: sqlite3.Connection,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    query = "SELECT * FROM market_digest_history"
    params: list = []
    clauses = []
    if start:
        clauses.append("date>=?"); params.append(start)
    if end:
        clauses.append("date<=?"); params.append(end)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY date"
    return pd.read_sql(query, conn, params=params)


# ─── Paper Portfolios ─────────────────────────────────────────────────────────

def create_paper_portfolio(
    conn: sqlite3.Connection,
    portfolio_id: str,
    strategy: str,
    starting_cash: float,
    display_name: Optional[str] = None,
    live_start_date: Optional[str] = None,
) -> None:
    today = date.today().isoformat()
    conn.execute(
        """INSERT INTO paper_portfolios
           (portfolio_id, display_name, strategy, created_date,
            live_start_date, starting_cash, current_cash)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(portfolio_id) DO NOTHING""",
        (portfolio_id, display_name or f"{strategy} portfolio",
         strategy, today, live_start_date, starting_cash, starting_cash),
    )


def get_paper_portfolio(conn: sqlite3.Connection, portfolio_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM paper_portfolios WHERE portfolio_id=?", (portfolio_id,)
    ).fetchone()
    return dict(row) if row else None


def get_all_paper_portfolios(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT * FROM paper_portfolios WHERE is_active=1 ORDER BY strategy",
        conn,
    )


def update_paper_cash(conn: sqlite3.Connection, portfolio_id: str, new_cash: float) -> None:
    conn.execute(
        "UPDATE paper_portfolios SET current_cash=? WHERE portfolio_id=?",
        (new_cash, portfolio_id),
    )


def upsert_paper_position(
    conn: sqlite3.Connection,
    portfolio_id: str,
    stock_id: int,
    shares: float,
    cost_basis: float,
    entry_date: str,
) -> None:
    """Open or update an open position (one open position per stock per portfolio)."""
    existing = conn.execute(
        """SELECT id FROM paper_positions
           WHERE portfolio_id=? AND stock_id=? AND is_open=1""",
        (portfolio_id, stock_id),
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE paper_positions SET shares=?, cost_basis=? WHERE id=?",
            (shares, cost_basis, existing["id"]),
        )
    else:
        conn.execute(
            """INSERT INTO paper_positions
               (portfolio_id, stock_id, shares, cost_basis, entry_date, is_open)
               VALUES (?,?,?,?,?,1)""",
            (portfolio_id, stock_id, shares, cost_basis, entry_date),
        )


def close_paper_position(conn: sqlite3.Connection, portfolio_id: str, stock_id: int) -> None:
    conn.execute(
        """UPDATE paper_positions SET is_open=0
           WHERE portfolio_id=? AND stock_id=? AND is_open=1""",
        (portfolio_id, stock_id),
    )


def get_open_paper_positions(conn: sqlite3.Connection, portfolio_id: str) -> pd.DataFrame:
    return pd.read_sql(
        """SELECT pp.*, s.ticker, s.name, s.sector
           FROM paper_positions pp
           JOIN stocks s ON s.id = pp.stock_id
           WHERE pp.portfolio_id=? AND pp.is_open=1""",
        conn, params=[portfolio_id],
    )


def log_paper_transaction(
    conn: sqlite3.Connection,
    portfolio_id: str,
    tx_date: str,
    tx_type: str,
    stock_id: int,
    shares: float,
    price: float,
    execution_price: float,
    cash_impact: float,
    reason: str,
    is_backsimulated: int = 0,
) -> None:
    conn.execute(
        """INSERT INTO paper_transactions
           (portfolio_id, date, type, stock_id, shares, price,
            execution_price, cash_impact, reason, is_backsimulated)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (portfolio_id, tx_date, tx_type, stock_id, shares, price,
         execution_price, cash_impact, reason, is_backsimulated),
    )


def get_paper_transactions(
    conn: sqlite3.Connection,
    portfolio_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    query = """
        SELECT pt.*, s.ticker, s.name
        FROM paper_transactions pt
        JOIN stocks s ON s.id = pt.stock_id
        WHERE pt.portfolio_id=?
    """
    params: list = [portfolio_id]
    if start:
        query += " AND pt.date>=?"; params.append(start)
    if end:
        query += " AND pt.date<=?"; params.append(end)
    query += " ORDER BY pt.date DESC"
    return pd.read_sql(query, conn, params=params)


# ─── Stock Notes ──────────────────────────────────────────────────────────────

def add_stock_note(
    conn: sqlite3.Connection,
    stock_id: int,
    content: str,
    note_type: str = "general",
) -> None:
    today = date.today().isoformat()
    conn.execute(
        """INSERT INTO stock_notes (stock_id, note_date, note_type, content, created_at)
           VALUES (?,?,?,?,?)""",
        (stock_id, today, note_type, content, today),
    )


def get_stock_notes(conn: sqlite3.Connection, stock_id: int) -> pd.DataFrame:
    return pd.read_sql(
        """SELECT * FROM stock_notes WHERE stock_id=?
           ORDER BY note_date DESC""",
        conn, params=[stock_id],
    )


def delete_stock_note(conn: sqlite3.Connection, note_id: int) -> None:
    conn.execute("DELETE FROM stock_notes WHERE id=?", (note_id,))
