"""
Horizon Ledger — SQLite Schema
Creates all nine core tables plus helper tables.
Uses WAL mode for better concurrent read performance.
"""

import sqlite3
import logging
from pathlib import Path
from config import DB_PATH

log = logging.getLogger(__name__)


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a connection with WAL mode, foreign keys, and row_factory."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


DDL_STATEMENTS = [
    # ── 1. stocks ─────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS stocks (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker       TEXT    UNIQUE NOT NULL,
        name         TEXT,
        sector       TEXT,
        industry     TEXT,
        cik          TEXT,
        market_cap   REAL,
        exchange     TEXT,
        is_active    INTEGER DEFAULT 1,
        added_date   TEXT,
        delisted_date TEXT
    )
    """,

    # ── 2. daily_prices ───────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS daily_prices (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_id  INTEGER NOT NULL REFERENCES stocks(id),
        date      TEXT    NOT NULL,
        open      REAL,
        high      REAL,
        low       REAL,
        close     REAL,
        adj_close REAL,
        volume    INTEGER,
        UNIQUE(stock_id, date)
    )
    """,

    # ── 3. fundamentals ───────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS fundamentals (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_id             INTEGER NOT NULL REFERENCES stocks(id),
        report_date          TEXT,
        filing_date          TEXT,
        fiscal_year          INTEGER,
        fiscal_quarter       INTEGER,
        revenue              REAL,
        gross_profit         REAL,
        ebit                 REAL,
        net_income           REAL,
        total_assets         REAL,
        total_debt           REAL,
        total_equity         REAL,
        current_assets       REAL,
        current_liabilities  REAL,
        cash                 REAL,
        operating_cash_flow  REAL,
        capex                REAL,
        free_cash_flow       REAL,
        dividends_paid       REAL,
        shares_outstanding   REAL,
        eps                  REAL,
        book_value_per_share REAL,
        data_source          TEXT DEFAULT 'edgar',
        UNIQUE(stock_id, report_date, fiscal_quarter)
    )
    """,

    # ── 4. scores ─────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS scores (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_id         INTEGER NOT NULL REFERENCES stocks(id),
        score_date       TEXT    NOT NULL,
        strategy         TEXT    NOT NULL,
        composite_score  REAL,
        score_components TEXT,
        weights_version  TEXT,
        UNIQUE(stock_id, score_date, strategy)
    )
    """,

    # ── 5. predictions ────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS predictions (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_id         INTEGER NOT NULL REFERENCES stocks(id),
        strategy         TEXT    NOT NULL,
        signal_date      TEXT    NOT NULL,
        composite_score  REAL,
        score_rank       INTEGER,
        score_components TEXT,
        return_5d        REAL,
        return_21d       REAL,
        return_63d       REAL,
        return_126d      REAL,
        return_252d      REAL,
        filled_date      TEXT,
        UNIQUE(stock_id, strategy, signal_date)
    )
    """,

    # ── 6. index_holdings ─────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS index_holdings (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        index_name   TEXT    NOT NULL,
        stock_id     INTEGER NOT NULL REFERENCES stocks(id),
        entry_date   TEXT,
        exit_date    TEXT,
        target_weight REAL,
        entry_price  REAL,
        exit_price   REAL,
        entry_score  REAL
    )
    """,

    # ── 7. rebalancing_history ────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS rebalancing_history (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        index_name     TEXT,
        rebalance_date TEXT,
        action         TEXT,
        stock_id       INTEGER REFERENCES stocks(id),
        old_weight     REAL,
        new_weight     REAL,
        reason         TEXT
    )
    """,

    # ── 8. weight_versions ────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS weight_versions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy    TEXT    NOT NULL,
        version_id  TEXT    NOT NULL,
        weights     TEXT    NOT NULL,
        created_date TEXT,
        approved_by  TEXT DEFAULT 'manual',
        notes        TEXT
    )
    """,

    # ── 9. reweighting_proposals ──────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS reweighting_proposals (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy                TEXT    NOT NULL,
        proposal_date           TEXT,
        current_weights         TEXT,
        proposed_weights        TEXT,
        ic_weights              TEXT,
        elastic_net_weights     TEXT,
        ensemble_weights        TEXT,
        confidence_intervals    TEXT,
        ic_summary              TEXT,
        walk_forward_efficiency REAL,
        estimated_turnover_impact REAL,
        status                  TEXT DEFAULT 'pending',
        approved_date           TEXT,
        approved_weights        TEXT,
        notes                   TEXT
    )
    """,

    # ── Supplementary: ticker_changes ─────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS ticker_changes (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        old_ticker TEXT,
        new_ticker TEXT,
        change_date TEXT,
        stock_id   INTEGER REFERENCES stocks(id)
    )
    """,

    # ── Supplementary: macro_data ─────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS macro_data (
        id     INTEGER PRIMARY KEY AUTOINCREMENT,
        series TEXT    NOT NULL,
        date   TEXT    NOT NULL,
        value  REAL,
        UNIQUE(series, date)
    )
    """,

    # ── Supplementary: regime_history ────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS regime_history (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        date         TEXT    NOT NULL UNIQUE,
        regime       TEXT,
        prob_bear    REAL,
        prob_neutral REAL,
        prob_bull    REAL
    )
    """,

    # ── Paper Trading: portfolios ─────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS paper_portfolios (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id     TEXT    UNIQUE NOT NULL,
        display_name     TEXT,
        strategy         TEXT    NOT NULL,
        created_date     TEXT,
        live_start_date  TEXT,
        starting_cash    REAL    NOT NULL,
        current_cash     REAL,
        rebalance_mode   TEXT    DEFAULT 'auto_mirror',
        is_active        INTEGER DEFAULT 1
    )
    """,

    # ── Paper Trading: open/closed positions ──────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS paper_positions (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id TEXT    REFERENCES paper_portfolios(portfolio_id),
        stock_id     INTEGER REFERENCES stocks(id),
        shares       REAL,
        cost_basis   REAL,
        entry_date   TEXT,
        is_open      INTEGER DEFAULT 1
    )
    """,

    # ── Paper Trading: full transaction ledger ────────────────────────────────
    # execution_price = price after slippage (price = pre-slippage reference price)
    """
    CREATE TABLE IF NOT EXISTS paper_transactions (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id      TEXT    REFERENCES paper_portfolios(portfolio_id),
        date              TEXT    NOT NULL,
        type              TEXT    NOT NULL,
        stock_id          INTEGER REFERENCES stocks(id),
        shares            REAL,
        price             REAL,
        execution_price   REAL,
        cash_impact       REAL,
        reason            TEXT,
        is_backsimulated  INTEGER DEFAULT 0
    )
    """,

    # ── Market Digest: daily health score snapshots ───────────────────────────
    """
    CREATE TABLE IF NOT EXISTS market_digest_history (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        date                  TEXT    NOT NULL UNIQUE,
        market_health_score   INTEGER,
        market_health_label   TEXT,
        regime                TEXT,
        cape_ratio            REAL,
        cape_percentile       REAL,
        yield_curve_slope     REAL,
        yield_curve_inverted  INTEGER,
        credit_spread         REAL,
        vix_level             REAL,
        sahm_rule_value       REAL,
        sahm_rule_triggered   INTEGER,
        tips_breakeven        REAL,
        pct_above_200sma      REAL,
        bubble_flags          TEXT,
        top25_consensus       TEXT,
        digest_text           TEXT
    )
    """,

    # ── Stock Notes: personal investment thesis and annotations ───────────────
    """
    CREATE TABLE IF NOT EXISTS stock_notes (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_id   INTEGER NOT NULL REFERENCES stocks(id),
        note_date  TEXT    NOT NULL,
        note_type  TEXT    DEFAULT 'general',
        content    TEXT    NOT NULL,
        created_at TEXT    NOT NULL
    )
    """,

    # ── Newsletter: issue history and send log ────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS newsletter_history (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        issue_date   TEXT    NOT NULL UNIQUE,
        issue_number INTEGER NOT NULL,
        pdf_path     TEXT,
        sent_to      INTEGER DEFAULT 0,
        sent_at      TEXT,
        status       TEXT    DEFAULT 'generated'
    )
    """,
]

INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_prices_stock_date ON daily_prices(stock_id, date)",
    "CREATE INDEX IF NOT EXISTS idx_fundamentals_stock_date ON fundamentals(stock_id, report_date)",
    "CREATE INDEX IF NOT EXISTS idx_scores_stock_strategy ON scores(stock_id, strategy, score_date)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_strategy_date ON predictions(strategy, signal_date)",
    "CREATE INDEX IF NOT EXISTS idx_holdings_index ON index_holdings(index_name, exit_date)",
    "CREATE INDEX IF NOT EXISTS idx_macro_series_date ON macro_data(series, date)",
    "CREATE INDEX IF NOT EXISTS idx_paper_tx_portfolio_date ON paper_transactions(portfolio_id, date)",
    "CREATE INDEX IF NOT EXISTS idx_paper_positions_portfolio ON paper_positions(portfolio_id, is_open)",
    "CREATE INDEX IF NOT EXISTS idx_digest_date ON market_digest_history(date)",
    "CREATE INDEX IF NOT EXISTS idx_stock_notes_stock ON stock_notes(stock_id, note_date)",
    "CREATE INDEX IF NOT EXISTS idx_newsletter_date ON newsletter_history(issue_date)",
]


def init_db(db_path: Path = DB_PATH) -> None:
    """Create all tables and indexes if they don't already exist."""
    conn = get_connection(db_path)
    with conn:
        for ddl in DDL_STATEMENTS:
            conn.execute(ddl)
        for idx in INDEX_STATEMENTS:
            conn.execute(idx)
    conn.close()
    log.info("Database initialized at %s", db_path)


def seed_initial_weights(conn: sqlite3.Connection) -> None:
    """Insert v1.0 factor weights for all four strategies if not present."""
    import json
    from datetime import date
    from config import STRATEGY_WEIGHTS, INITIAL_WEIGHT_VERSION

    today = date.today().isoformat()
    for strategy, weights in STRATEGY_WEIGHTS.items():
        existing = conn.execute(
            "SELECT id FROM weight_versions WHERE strategy=? AND version_id=?",
            (strategy, INITIAL_WEIGHT_VERSION),
        ).fetchone()
        if not existing:
            conn.execute(
                """INSERT INTO weight_versions
                   (strategy, version_id, weights, created_date, approved_by, notes)
                   VALUES (?,?,?,?,?,?)""",
                (
                    strategy,
                    INITIAL_WEIGHT_VERSION,
                    json.dumps(weights),
                    today,
                    "initial_seed",
                    "Default weights from config.py",
                ),
            )
    conn.commit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    conn = get_connection()
    seed_initial_weights(conn)
    conn.close()
    print(f"Database ready at {DB_PATH}")
