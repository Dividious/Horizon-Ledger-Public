"""
Horizon Ledger — Rules-Based Index Construction
Builds and maintains four strategy indexes with entry/exit banding.

Entry/exit banding reduces unnecessary turnover:
  - Enter if rank ≤ N
  - Exit if rank > N * (1 + EXIT_BUFFER) = N * 1.4 (for 40% buffer)

Sector cap: max 25% per GICS sector.
Weighting: equal weight (default) or inverse volatility.
"""

import json
import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    INDEX_SIZES,
    INDEX_EXIT_BUFFER,
    SECTOR_MAX_WEIGHT,
    TURNAROUND_MAX_POSITION,
    TURNAROUND_TOTAL_MAX,
)
from db.schema import get_connection
from db.queries import (
    get_latest_scores,
    get_current_holdings,
    add_holding,
    close_holding,
    get_price_on_date,
    get_stock_id,
)

log = logging.getLogger(__name__)

INDEX_NAMES = {
    "long_term":    "long_term_index",
    "dividend":     "dividend_index",
    "turnaround":   "turnaround_index",
    "swing":        "swing_index",
    # Public-facing indexes
    "conservative": "conservative_index",
    "aggressive":   "aggressive_index",
}


def _get_vol_and_score(tickers, strategy, conn, as_of):
    """Helper: returns (vols dict, scores dict) for tickers."""
    from db.queries import get_prices, get_latest_scores
    vols, scores = {}, {}
    scores_df = get_latest_scores(conn, strategy)
    score_map = {}
    if not scores_df.empty and "ticker" in scores_df.columns:
        score_map = dict(zip(scores_df["ticker"], scores_df["composite_score"]))
    for tkr in tickers:
        sid = get_stock_id(conn, tkr)
        if sid is None:
            vols[tkr] = 1.0
            scores[tkr] = 50.0
            continue
        df = get_prices(
            conn, sid,
            start=(date.fromisoformat(as_of) - timedelta(days=60)).isoformat(),
            end=as_of,
        )
        if df.empty or len(df) < 10:
            vols[tkr] = 1.0
        else:
            returns = df["adj_close"].pct_change().dropna()
            vol = returns.tail(21).std() * np.sqrt(252)
            vols[tkr] = max(vol, 0.001)
        scores[tkr] = float(score_map.get(tkr, 50.0))
    return vols, scores


def compute_index_weights(
    tickers: list[str],
    strategy: str,
    conn,
    as_of: str,
    method: str = "equal",
) -> dict[str, float]:
    """
    Compute target weights for a set of tickers.

    method:
      'equal'      — 1/N equal weight
      'inv_vol'    — Inverse 21-day realized volatility
      'half_kelly' — Half-Kelly sizing using composite score as expected-return proxy
                     w_i ∝ 0.5 × (mu_i / vol_i²)  where mu_i = score_i / 100 × scale
    """
    n = len(tickers)
    if n == 0:
        return {}

    if method == "inv_vol":
        vols, _ = _get_vol_and_score(tickers, strategy, conn, as_of)
        inv_vols = {t: 1 / vols[t] for t in tickers}
        total = sum(inv_vols.values())
        return {t: v / total for t, v in inv_vols.items()}

    elif method == "half_kelly":
        vols, scores = _get_vol_and_score(tickers, strategy, conn, as_of)
        # Normalize scores to [0, 1] as expected-return proxy
        max_s = max(scores.values()) if scores else 100.0
        min_s = min(scores.values()) if scores else 0.0
        rng = max(max_s - min_s, 1.0)
        kelly_weights = {}
        for tkr in tickers:
            mu = (scores[tkr] - min_s) / rng   # 0-1 normalized
            vol2 = vols[tkr] ** 2
            kelly_weights[tkr] = 0.5 * (mu / vol2) if vol2 > 0 else 0.5 / n
        total = sum(kelly_weights.values())
        if total <= 0:
            return {t: 1 / n for t in tickers}
        return {t: v / total for t, v in kelly_weights.items()}

    else:
        # Default: equal weight
        return {t: 1 / n for t in tickers}


def apply_sector_cap(
    weights: dict[str, float],
    sectors: dict[str, str],
    max_sector_weight: float = SECTOR_MAX_WEIGHT,
) -> dict[str, float]:
    """
    Enforce sector cap: redistribute excess weight proportionally.
    Iterates until all sectors are within cap.
    """
    for _ in range(10):   # Max 10 iterations
        # Sum weights by sector
        sector_weights: dict[str, float] = {}
        for tkr, w in weights.items():
            s = sectors.get(tkr, "Unknown")
            sector_weights[s] = sector_weights.get(s, 0) + w

        capped = {s: w for s, w in sector_weights.items() if w > max_sector_weight}
        if not capped:
            break

        # Reduce over-weight sectors
        for sector, total in capped.items():
            cap_ratio = max_sector_weight / total
            sector_tickers = [t for t in weights if sectors.get(t) == sector]
            for tkr in sector_tickers:
                weights[tkr] = weights[tkr] * cap_ratio

        # Re-normalize to sum to 1
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {t: w / total_w for t, w in weights.items()}

    return weights


def reconstitute_index(
    strategy: str,
    as_of: Optional[str] = None,
    weighting: str = "equal",
    dry_run: bool = False,
) -> dict:
    """
    Full index reconstitution:
      1. Get latest scores for strategy
      2. Apply entry/exit banding against current holdings
      3. Apply sector cap
      4. Generate add/remove/weight_change actions
      5. Persist to DB (unless dry_run)

    Returns a dict with: adds, removes, weight_changes, proposed_holdings
    """
    if as_of is None:
        as_of = date.today().isoformat()

    index_name = INDEX_NAMES.get(strategy, f"{strategy}_index")
    target_size = INDEX_SIZES.get(strategy, 25)
    exit_threshold = int(target_size * (1 + INDEX_EXIT_BUFFER))

    conn = get_connection()

    # Get latest scores
    scores_df = get_latest_scores(conn, strategy)
    if scores_df.empty:
        conn.close()
        log.warning("No scores available for %s", strategy)
        return {"adds": [], "removes": [], "weight_changes": [], "proposed_holdings": []}

    # Assign ranks (already sorted by composite_score DESC)
    scores_df = scores_df.reset_index(drop=True)
    scores_df["rank"] = range(1, len(scores_df) + 1)

    # Current holdings
    current_df = get_current_holdings(conn, index_name)
    current_tickers = set(current_df["ticker"].tolist()) if not current_df.empty else set()

    # Determine in/out based on banding
    new_entries = scores_df[scores_df["rank"] <= target_size]["ticker"].tolist()
    retain_if_in = scores_df[scores_df["rank"] <= exit_threshold]["ticker"].tolist()

    # Stocks to add: in new_entries but not currently held
    to_add = [t for t in new_entries if t not in current_tickers]

    # Stocks to remove: currently held but rank > exit_threshold
    to_remove = [t for t in current_tickers if t not in retain_if_in]

    # Final proposed holdings
    proposed = [t for t in retain_if_in if t in current_tickers or t in new_entries]
    proposed = list(dict.fromkeys(proposed))  # Deduplicate preserving order

    # Sector cap
    sector_map = dict(zip(scores_df["ticker"], scores_df["sector"].fillna("Unknown")))
    raw_weights = compute_index_weights(proposed, strategy, conn, as_of, weighting)
    capped_weights = apply_sector_cap(raw_weights, sector_map)

    # Build action list
    adds, removes, weight_changes = [], [], []

    # Removes
    if not current_df.empty:
        for _, h in current_df.iterrows():
            tkr = h["ticker"]
            if tkr in to_remove:
                price = get_price_on_date(conn, h["stock_id"], as_of)
                removes.append({
                    "ticker": tkr,
                    "stock_id": h["stock_id"],
                    "holding_id": h["id"],
                    "exit_price": price,
                    "reason": f"Rank > exit threshold ({exit_threshold})",
                })
            elif tkr in proposed:
                new_w = capped_weights.get(tkr, 0.0)
                old_w = h.get("target_weight", 0.0)
                if abs(new_w - old_w) > 0.005:
                    weight_changes.append({
                        "ticker": tkr,
                        "old_weight": old_w,
                        "new_weight": new_w,
                    })

    # Adds
    for tkr in to_add:
        sid_row = scores_df[scores_df["ticker"] == tkr]
        sid = int(sid_row.iloc[0]["stock_id"]) if not sid_row.empty else None
        if sid is None:
            continue
        price = get_price_on_date(conn, sid, as_of)
        score = float(sid_row.iloc[0]["composite_score"]) if not sid_row.empty else None
        adds.append({
            "ticker": tkr,
            "stock_id": sid,
            "entry_price": price,
            "target_weight": capped_weights.get(tkr, 0.0),
            "entry_score": score,
        })

    result = {
        "strategy":           strategy,
        "index_name":         index_name,
        "as_of":              as_of,
        "adds":               adds,
        "removes":            removes,
        "weight_changes":     weight_changes,
        "proposed_holdings":  [{"ticker": t, "weight": capped_weights.get(t, 0.0)} for t in proposed],
        "target_size":        target_size,
        "actual_size":        len(proposed),
    }

    if not dry_run:
        _apply_reconstitution(conn, index_name, as_of, adds, removes, weight_changes)

    conn.close()
    log.info(
        "Index reconstitution for %s: +%d -%d ~%d weight changes",
        strategy, len(adds), len(removes), len(weight_changes),
    )
    return result


def _apply_reconstitution(conn, index_name, as_of, adds, removes, weight_changes):
    """Write the reconstitution to the database."""
    from db.queries import close_holding

    with conn:
        # Close removed holdings
        for r in removes:
            close_holding(conn, r["holding_id"], as_of, r.get("exit_price"))
            conn.execute(
                """INSERT INTO rebalancing_history
                   (index_name,rebalance_date,action,stock_id,old_weight,new_weight,reason)
                   VALUES (?,?,?,?,?,?,?)""",
                (index_name, as_of, "REMOVE", r["stock_id"],
                 0.0, 0.0, r["reason"]),
            )

        # Add new holdings
        for a in adds:
            add_holding(
                conn, index_name, a["stock_id"], as_of,
                a["target_weight"], a.get("entry_price"), a.get("entry_score"),
            )
            conn.execute(
                """INSERT INTO rebalancing_history
                   (index_name,rebalance_date,action,stock_id,old_weight,new_weight,reason)
                   VALUES (?,?,?,?,?,?,?)""",
                (index_name, as_of, "ADD", a["stock_id"],
                 0.0, a["target_weight"], "New entrant"),
            )

        # Update weight changes
        for wc in weight_changes:
            tkr = wc["ticker"]
            conn.execute(
                """UPDATE index_holdings SET target_weight=?
                   WHERE index_name=? AND exit_date IS NULL
                   AND stock_id=(SELECT id FROM stocks WHERE ticker=?)""",
                (wc["new_weight"], index_name, tkr),
            )
            sid_row = conn.execute("SELECT id FROM stocks WHERE ticker=?", (tkr,)).fetchone()
            if sid_row:
                conn.execute(
                    """INSERT INTO rebalancing_history
                       (index_name,rebalance_date,action,stock_id,old_weight,new_weight,reason)
                       VALUES (?,?,?,?,?,?,?)""",
                    (index_name, as_of, "WEIGHT_CHANGE", sid_row["id"],
                     wc["old_weight"], wc["new_weight"], "Reconstitution reweight"),
                )
        conn.commit()
