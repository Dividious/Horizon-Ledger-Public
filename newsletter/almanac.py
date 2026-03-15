"""
Horizon Ledger — Newsletter Almanac Data
Historical market statistics, seasonal patterns, and economic calendar.

NOT FINANCIAL ADVICE. Historical patterns do not guarantee future results.
"""

from datetime import date, timedelta
from typing import Optional


# ─── Historical Monthly Stats (S&P 500, empirical 1950–2024) ─────────────────

MONTHLY_STATS = {
    1:  {"name": "January",   "positive_pct": 62, "avg_return": 1.1,  "best": 13.2,  "worst": -8.7},
    2:  {"name": "February",  "positive_pct": 57, "avg_return": 0.1,  "best": 10.7,  "worst": -11.0},
    3:  {"name": "March",     "positive_pct": 65, "avg_return": 1.2,  "best": 9.7,   "worst": -9.9},
    4:  {"name": "April",     "positive_pct": 68, "avg_return": 1.5,  "best": 12.7,  "worst": -9.7},
    5:  {"name": "May",       "positive_pct": 58, "avg_return": 0.3,  "best": 9.2,   "worst": -8.1},
    6:  {"name": "June",      "positive_pct": 56, "avg_return": 0.0,  "best": 7.6,   "worst": -8.4},
    7:  {"name": "July",      "positive_pct": 63, "avg_return": 1.0,  "best": 8.5,   "worst": -7.8},
    8:  {"name": "August",    "positive_pct": 56, "avg_return": 0.1,  "best": 11.6,  "worst": -14.5},
    9:  {"name": "September", "positive_pct": 45, "avg_return": -0.7, "best": 8.8,   "worst": -11.0},
    10: {"name": "October",   "positive_pct": 63, "avg_return": 0.9,  "best": 10.8,  "worst": -21.8},
    11: {"name": "November",  "positive_pct": 70, "avg_return": 1.7,  "best": 10.2,  "worst": -8.4},
    12: {"name": "December",  "positive_pct": 74, "avg_return": 1.6,  "best": 11.2,  "worst": -6.0},
}


# ─── Sector Seasonality by Month ─────────────────────────────────────────────

SECTOR_SEASONALITY = {
    1:  {"favorable": ["Technology", "Consumer Discretionary", "Health Care"], "weak": ["Energy", "Utilities"]},
    2:  {"favorable": ["Health Care", "Consumer Staples"], "weak": ["Energy", "Materials"]},
    3:  {"favorable": ["Energy", "Industrials", "Materials"], "weak": ["Utilities", "Real Estate"]},
    4:  {"favorable": ["Technology", "Consumer Discretionary", "Industrials"], "weak": ["Utilities"]},
    5:  {"favorable": ["Health Care", "Consumer Staples"], "weak": ["Technology", "Energy"]},
    6:  {"favorable": ["Energy", "Consumer Discretionary"], "weak": ["Utilities", "Real Estate"]},
    7:  {"favorable": ["Technology", "Consumer Discretionary", "Financials"], "weak": ["Energy"]},
    8:  {"favorable": ["Utilities", "Consumer Staples"], "weak": ["Technology", "Financials"]},
    9:  {"favorable": ["Health Care", "Consumer Staples", "Utilities"], "weak": ["Technology", "Consumer Discretionary"]},
    10: {"favorable": ["Technology", "Consumer Discretionary", "Financials"], "weak": ["Utilities", "Real Estate"]},
    11: {"favorable": ["Consumer Discretionary", "Technology", "Industrials"], "weak": ["Energy", "Materials"]},
    12: {"favorable": ["Consumer Discretionary", "Technology", "Real Estate"], "weak": ["Energy", "Health Care"]},
}


# ─── FOMC Meeting End Dates (hardcoded 2026–2027) ────────────────────────────

FED_MEETING_DATES = [
    "2026-01-29", "2026-03-19", "2026-05-07", "2026-06-18",
    "2026-07-30", "2026-09-17", "2026-10-29", "2026-12-10",
    "2027-01-28", "2027-03-18", "2027-05-06", "2027-06-17",
    "2027-07-29", "2027-09-16", "2027-10-28", "2027-12-09",
]

# Earnings season windows as (month, day) tuples: (start_month, start_day, end_month, end_day)
EARNINGS_WINDOWS = [
    (1, 15, 2, 15),
    (4, 15, 5, 15),
    (7, 15, 8, 15),
    (10, 15, 11, 15),
]


# ─── Helper Functions ─────────────────────────────────────────────────────────

def find_nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """
    Return the nth occurrence of a weekday in a given year/month.

    Parameters
    ----------
    year    : int
    month   : int
    weekday : int  (0=Monday, 1=Tuesday, ..., 4=Friday, 5=Saturday, 6=Sunday)
    n       : int  (1-based: 1=first, 2=second, etc.)

    Returns
    -------
    date
    """
    first_of_month = date(year, month, 1)
    # How many days until the target weekday from the 1st?
    days_ahead = weekday - first_of_month.weekday()
    if days_ahead < 0:
        days_ahead += 7
    first_occurrence = first_of_month + timedelta(days=days_ahead)
    return first_occurrence + timedelta(weeks=(n - 1))


def _in_earnings_season(d: date) -> bool:
    """Return True if the date falls within an earnings season window."""
    for (sm, sd, em, ed) in EARNINGS_WINDOWS:
        start = date(d.year, sm, sd)
        # Handle cross-year windows (none currently, but be safe)
        if em < sm:
            end = date(d.year + 1, em, ed)
        else:
            end = date(d.year, em, ed)
        if start <= d <= end:
            return True
    return False


# ─── Public API ──────────────────────────────────────────────────────────────

def get_monthly_stats(month: int) -> dict:
    """
    Return historical S&P 500 statistics for a given month (1–12).

    Keys: name, positive_pct, avg_return, best, worst
    Source: Empirical data 1950–2024.
    """
    if month not in MONTHLY_STATS:
        raise ValueError(f"month must be 1–12, got {month!r}")
    return dict(MONTHLY_STATS[month])


def get_sector_seasonality(month: int) -> dict:
    """
    Return seasonal sector tendencies for a given month (1–12).

    Returns
    -------
    dict with keys "favorable" and "weak", each a list of sector name strings.
    """
    if month not in SECTOR_SEASONALITY:
        raise ValueError(f"month must be 1–12, got {month!r}")
    data = SECTOR_SEASONALITY[month]
    return {
        "favorable": list(data["favorable"]),
        "weak": list(data["weak"]),
    }


def get_economic_calendar(start_date: str, days: int = 35) -> list:
    """
    Return a list of upcoming economic events within *days* calendar days
    of start_date (inclusive).

    Parameters
    ----------
    start_date : str   ISO format "YYYY-MM-DD"
    days       : int   window length (default 35)

    Returns
    -------
    list of dicts: {"date": "YYYY-MM-DD", "event": str, "why_it_matters": str}
    Sorted ascending by date; duplicates (same date + event) de-duped.
    """
    start = date.fromisoformat(start_date)
    end = start + timedelta(days=days)

    events: list[dict] = []
    seen: set[tuple] = set()

    def _add(d: date, event: str, why: str) -> None:
        if start <= d <= end:
            key = (d.isoformat(), event)
            if key not in seen:
                seen.add(key)
                events.append({"date": d.isoformat(), "event": event, "why_it_matters": why})

    # Iterate over each month that overlaps with the window
    months_to_check: set[tuple] = set()
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        months_to_check.add((cursor.year, cursor.month))
        # Advance to next month
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)

    for year, month in sorted(months_to_check):
        # ── NFP (Jobs Report): first Friday of each month ─────────────────
        try:
            nfp_date = find_nth_weekday(year, month, 4, 1)  # 4=Friday
            _add(
                nfp_date,
                "NFP Jobs Report",
                "The monthly payrolls number is the single most market-moving "
                "economic release — it shapes Fed rate expectations and risk sentiment.",
            )
        except ValueError:
            pass

        # ── CPI Release: approximately 2nd Tuesday of each month ──────────
        try:
            cpi_date = find_nth_weekday(year, month, 1, 2)  # 1=Tuesday, 2nd occurrence
            _add(
                cpi_date,
                "CPI Inflation Report",
                "The Consumer Price Index directly influences Federal Reserve "
                "rate decisions and bond yields, which ripple through all asset classes.",
            )
        except ValueError:
            pass

    # ── Fed Meeting End Dates ──────────────────────────────────────────────
    for ds in FED_MEETING_DATES:
        fed_date = date.fromisoformat(ds)
        _add(
            fed_date,
            "FOMC Rate Decision",
            "The Federal Reserve announces its interest-rate decision and "
            "publishes updated economic projections. Market volatility typically "
            "spikes around this event.",
        )

    # ── Earnings Season Note ───────────────────────────────────────────────
    # Add a single note at the start of the window if we are in an earnings season
    # or if one begins within the window.
    for day_offset in range(days + 1):
        check_day = start + timedelta(days=day_offset)
        if _in_earnings_season(check_day):
            # Find the window start that encompasses check_day
            for (sm, sd, em, ed) in EARNINGS_WINDOWS:
                season_start = date(check_day.year, sm, sd)
                season_end_year = check_day.year if em >= sm else check_day.year + 1
                season_end = date(season_end_year, em, ed)
                if season_start <= check_day <= season_end:
                    # Add a note on the season start date (if in window)
                    _add(
                        season_start,
                        "Earnings Season (begins)",
                        "Major-company quarterly earnings reports begin. "
                        "Expect elevated single-stock volatility and potential "
                        "sector-level price moves driven by guidance revisions.",
                    )
                    break
            break  # Only add one earnings season note

    events.sort(key=lambda e: e["date"])
    return events


def get_cycle_position(market_health_row: dict) -> str:
    """
    Return a 2–3 sentence plain-English description of current market cycle position.

    Parameters
    ----------
    market_health_row : dict
        A row from market_digest_history (or equivalent dict) containing:
            regime               : str   "bull" / "neutral" / "bear"
            cape_percentile      : float  0–100
            yield_curve_inverted : int    0 or 1
            yield_curve_slope    : float  percentage points (optional, for un-inversion check)
            sahm_rule_triggered  : int    0 or 1
    """
    regime = (market_health_row.get("regime") or "neutral").lower()
    cape_pct = float(market_health_row.get("cape_percentile") or 50)
    yield_inverted = bool(market_health_row.get("yield_curve_inverted", 0))
    slope = market_health_row.get("yield_curve_slope")  # may be None
    sahm_triggered = bool(market_health_row.get("sahm_rule_triggered", 0))

    # ── Primary cycle sentence ────────────────────────────────────────────
    if regime == "bull" and cape_pct < 70:
        cycle = "Mid-cycle bull market with reasonable valuations."
    elif regime == "bull" and cape_pct >= 70:
        cycle = "Late-cycle bull market with elevated valuations."
    elif regime == "neutral":
        cycle = "Transitional market environment — neither clearly bullish nor bearish."
    else:
        cycle = "Bear market or high-uncertainty environment."

    # ── Yield curve addendum ──────────────────────────────────────────────
    if yield_inverted:
        cycle += (
            " The yield curve remains inverted, historically a leading recession indicator."
        )
    elif slope is not None and slope > 0:
        # Positive slope now — may have recently normalized (we infer from context)
        cycle += " The yield curve has recently normalized."

    # ── Sahm Rule addendum ────────────────────────────────────────────────
    if sahm_triggered:
        cycle += " The Sahm Rule recession indicator is currently active."

    return cycle


def get_cape_10yr_outlook(cape_ratio: float) -> dict:
    """
    Campbell-Shiller 10-year annualized return forecast based on current CAPE.

    Returns
    -------
    dict: {"median": float, "p25": float, "p75": float}
          All values are annualized percentage returns.

    Note: This is a historically-derived heuristic, not a guarantee.
    """
    if cape_ratio < 15:
        return {"median": 12.0, "p25": 8.0,  "p75": 15.0}
    elif cape_ratio < 20:
        return {"median": 9.0,  "p25": 5.0,  "p75": 13.0}
    elif cape_ratio < 25:
        return {"median": 6.5,  "p25": 3.0,  "p75": 10.0}
    elif cape_ratio < 30:
        return {"median": 4.5,  "p25": 1.0,  "p75": 8.0}
    elif cape_ratio < 35:
        return {"median": 2.5,  "p25": -1.0, "p75": 6.0}
    else:
        return {"median": 1.0,  "p25": -3.0, "p75": 4.5}
