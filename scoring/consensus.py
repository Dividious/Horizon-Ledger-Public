"""
Horizon Ledger — Cross-Strategy Consensus Top 25
Identifies stocks that rank highly across multiple scoring strategies.

Algorithm:
  1. For each stock in the active universe, retrieve its most recent
     composite score for each strategy it qualifies for (not excluded by
     hard filters).
  2. Convert each composite score to a percentile rank within that
     strategy's scored universe.
  3. Average the percentile ranks across qualifying strategies.
  4. Weight by strategy count: multiply by sqrt(n_strategies) so that
     a stock qualifying in 3 strategies outranks one qualifying in 1
     strategy with the same average percentile.
  5. Only include stocks qualifying in at least 2 strategies.
  6. Return top 25 by consensus score.

NOT FINANCIAL ADVICE.
"""

import json
import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from db.schema import get_connection
from db.queries import get_active_universe, upsert_digest, get_latest_digest

log = logging.getLogger(__name__)

STRATEGIES = ["long_term", "dividend", "turnaround", "swing"]


def compute_consensus_top25(
    as_of: Optional[str] = None,
    min_strategies: int = 2,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Compute the cross-strategy consensus top-25 stock list.

    Returns DataFrame with columns:
      ticker, name, sector, consensus_score, n_strategies,
      long_term_score, dividend_score, turnaround_score, swing_score,
      long_term_pct, dividend_pct, turnaround_pct, swing_pct
    """
    if as_of is None:
        as_of = date.today().isoformat()

    conn = get_connection()
    universe = get_active_universe(conn)
    if universe.empty:
        conn.close()
        return pd.DataFrame()

    # ── Fetch latest scores for every strategy ────────────────────────────────
    strategy_scores: dict[str, pd.DataFrame] = {}
    for strategy in STRATEGIES:
        df = pd.read_sql(
            """
            SELECT sc.stock_id, sc.composite_score, s.ticker
            FROM scores sc
            JOIN stocks s ON s.id = sc.stock_id
            WHERE sc.strategy = ?
              AND sc.score_date = (
                  SELECT MAX(score_date) FROM scores
                  WHERE strategy = ? AND stock_id = sc.stock_id
                    AND score_date <= ?
              )
            """,
            conn,
            params=[strategy, strategy, as_of],
        )
        if not df.empty:
            # Cross-sectional percentile rank within this strategy's scored universe
            df["pct_rank"] = df["composite_score"].rank(pct=True) * 100
            strategy_scores[strategy] = df.set_index("stock_id")

    conn.close()

    if not strategy_scores:
        return pd.DataFrame()

    # ── Build consensus score per stock ──────────────────────────────────────
    rows = []
    for _, stock in universe.iterrows():
        sid     = stock["id"]
        ticker  = stock["ticker"]
        name    = stock.get("name", "")
        sector  = stock.get("sector", "")

        qualifying_pcts   = []
        strategy_raw      = {}
        strategy_pct      = {}

        for strategy, sdf in strategy_scores.items():
            if sid in sdf.index:
                row_data = sdf.loc[sid]
                raw   = float(row_data["composite_score"])
                pct   = float(row_data["pct_rank"])
                qualifying_pcts.append(pct)
                strategy_raw[strategy] = round(raw, 1)
                strategy_pct[strategy] = round(pct, 1)

        n = len(qualifying_pcts)
        if n < min_strategies:
            continue

        avg_pct        = np.mean(qualifying_pcts)
        # Multiply by sqrt(n) to reward multi-strategy qualification
        consensus_score = avg_pct * np.sqrt(n)

        rows.append({
            "ticker":            ticker,
            "name":              name,
            "sector":            sector,
            "consensus_score":   round(consensus_score, 2),
            "n_strategies":      n,
            "avg_pct_rank":      round(avg_pct, 1),
            "long_term_score":   strategy_raw.get("long_term"),
            "dividend_score":    strategy_raw.get("dividend"),
            "turnaround_score":  strategy_raw.get("turnaround"),
            "swing_score":       strategy_raw.get("swing"),
            "long_term_pct":     strategy_pct.get("long_term"),
            "dividend_pct":      strategy_pct.get("dividend"),
            "turnaround_pct":    strategy_pct.get("turnaround"),
            "swing_pct":         strategy_pct.get("swing"),
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("consensus_score", ascending=False).head(top_n)
    result = result.reset_index(drop=True)
    result.index = result.index + 1  # 1-based rank
    result.index.name = "rank"
    return result.reset_index()


def store_consensus_in_digest(as_of: Optional[str] = None) -> None:
    """
    Compute consensus top-25 and store as JSON in market_digest_history.top25_consensus.
    If no digest row exists for today, creates one with just the consensus field.
    """
    if as_of is None:
        as_of = date.today().isoformat()

    df = compute_consensus_top25(as_of=as_of)
    if df.empty:
        log.info("Consensus top-25: no qualifying stocks found")
        return

    tickers_json = json.dumps(df[["rank", "ticker", "consensus_score"]].to_dict("records"))

    conn = get_connection()
    with conn:
        existing = conn.execute(
            "SELECT id FROM market_digest_history WHERE date=?", (as_of,)
        ).fetchone()

        if existing:
            conn.execute(
                "UPDATE market_digest_history SET top25_consensus=? WHERE date=?",
                (tickers_json, as_of),
            )
        else:
            conn.execute(
                """INSERT INTO market_digest_history (date, top25_consensus)
                   VALUES (?, ?)
                   ON CONFLICT(date) DO UPDATE SET top25_consensus=excluded.top25_consensus""",
                (as_of, tickers_json),
            )
        conn.commit()
    conn.close()
    log.info("Consensus top-25 stored: %d stocks", len(df))
