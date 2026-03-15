"""
Horizon Ledger — Central Configuration
All constants, weight tables, thresholds, and free-data API settings.
NOT FINANCIAL ADVICE.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
DB_DIR      = BASE_DIR / "db"
SECRETS_DIR = BASE_DIR / "secrets"
DOCS_DIR    = BASE_DIR / "docs"
NEWSLETTER_DIR = DATA_DIR / "newsletters"

DB_PATH = DB_DIR / "horizon_ledger.db"

DATA_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)
SECRETS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)
NEWSLETTER_DIR.mkdir(exist_ok=True)

# ─── FRED API ─────────────────────────────────────────────────────────────────
# Store your FRED key in secrets/fred_api_key.txt  OR  as env var FRED_API_KEY
_fred_key_file = SECRETS_DIR / "fred_api_key.txt"
if _fred_key_file.exists():
    FRED_API_KEY = _fred_key_file.read_text().strip()
else:
    FRED_API_KEY = os.environ.get("FRED_API_KEY", "ab0fd2cf17b4080b3f9fe275c73517a5")

FRED_SERIES = [
    "DGS10", "DGS2", "VIXCLS", "BAMLH0A0HYM2", "CPIAUCSL",
    "SAHMREALTIME",   # Sahm Rule Realtime indicator
    "T10YIE",         # 10-Year Breakeven Inflation Rate (TIPS spread)
]
RISK_FREE_TICKER = "DGS10"   # 10-year Treasury from FRED (annualized %)

# ─── Universe Filters ─────────────────────────────────────────────────────────
UNIVERSE_MIN_MARKET_CAP  = 100_000_000   # $100 M
UNIVERSE_MIN_AVG_VOLUME  = 1_000_000     # $1 M notional daily average (20-day)
UNIVERSE_MIN_LISTING_AGE_DAYS = 365

# ─── Scoring ──────────────────────────────────────────────────────────────────
SCORING_UNIVERSE_MIN_STOCKS = 50
MISSING_FACTOR_FILL = 50.0     # Neutral percentile for stocks with missing factor data

# ─── Index Sizes & Rebalancing ───────────────────────────────────────────────
INDEX_SIZES = {
    "long_term":    25,
    "dividend":     25,
    "turnaround":   15,
    "swing":        10,
    # Public-facing indexes
    "conservative": 25,   # Quality + Value + Low-Vol + Income; quarterly rebalance
    "aggressive":   25,   # Momentum + Growth + Quality; monthly rebalance
}
INDEX_EXIT_BUFFER        = 0.40   # Exit at rank > size * (1 + buffer)
SECTOR_MAX_WEIGHT        = 0.25   # 25% max per GICS sector per index
REBALANCE_DRIFT_THRESHOLD = 0.05  # 5% drift from target triggers a flag

# ─── Reweighting Guardrails ──────────────────────────────────────────────────
REWEIGHTING_MIN_OBSERVATIONS      = 60    # Monthly predictions required before reweighting
REWEIGHTING_MAX_CHANGE_PER_CYCLE  = 0.05  # Max ±5 pp per factor per cycle
REWEIGHTING_MAX_SINGLE_FACTOR     = 0.40  # No factor can exceed 40%
REWEIGHTING_MIN_SINGLE_FACTOR     = 0.05  # If included, minimum 5%
REWEIGHTING_AUTO_APPROVE_THRESHOLD = 0.02  # Auto-approve if all changes < 2%
IC_MIN_USEFUL     = 0.01    # Factors below this IC are flagged for review
WFE_MIN_ACCEPTABLE = 0.50  # Walk-Forward Efficiency: OOS/IS ratio lower bound
T_STAT_THRESHOLD  = 3.0    # Harvey-Liu-Zhu (2016) t-stat threshold for factor validity

# ─── Data Fetching ────────────────────────────────────────────────────────────
PRICE_HISTORY_YEARS       = 5
EDGAR_FUNDAMENTALS_LAG_DAYS = 45   # Use fundamentals only if filing_date <= as_of_date - lag

# SEC EDGAR bulk data URLs
EDGAR_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/"
EDGAR_SUBMISSIONS_URL  = "https://data.sec.gov/submissions/"
EDGAR_BULK_ZIP_URL     = "https://data.sec.gov/Archives/edgar/full-index/xbrl/"

# Wikipedia S&P 500 list
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# iShares Russell 1000 ETF holdings CSV
RUSSELL1000_URL = (
    "https://www.ishares.com/us/products/239707/"
    "ishares-russell-1000-etf/1467271812596.ajax"
    "?fileType=csv&fileName=IWB_holdings&dataType=fund"
)

# Fama-French factor data
FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)

# ─── Schedule (documentation only — use cron/systemd for actual scheduling) ───
DAILY_RUN_TIME    = "18:30"   # 6:30 PM ET
WEEKLY_RUN_DAY    = "Saturday"
QUARTERLY_MONTHS  = [1, 4, 7, 10]

# ─── Factor Weights by Strategy ──────────────────────────────────────────────
# Weights must sum to 1.0.  These are the initial/default weights (v1.0).
# Updated versions are stored in the weight_versions table after human approval.

LONG_TERM_WEIGHTS = {
    "gross_profitability":  0.15,
    "roic":                 0.10,
    "piotroski_f":          0.10,
    "earnings_yield":       0.10,
    "fcf_yield":            0.08,
    "ev_to_ebitda_inv":     0.07,
    "price_to_sales_inv":   0.05,
    "price_to_book_inv":    0.05,
    "altman_z":             0.05,
    "debt_to_equity_inv":   0.05,
    "current_ratio":        0.05,
    "revenue_cagr_5y":      0.05,
    "earnings_cagr_5y":     0.05,
    "momentum_6m":          0.05,
}

DIVIDEND_WEIGHTS = {
    "altman_z_safety":         0.10,
    "fcf_payout_ratio_inv":    0.08,
    "debt_to_equity_inv":      0.07,
    "earnings_payout_inv":     0.05,
    "div_growth_5y":           0.10,
    "div_growth_10y":          0.08,
    "consecutive_increases":   0.07,
    "dividend_yield":          0.10,
    "shareholder_yield":       0.05,
    "pe_ratio_inv":            0.05,
    "p_fcf_inv":               0.05,
    "ev_ebitda_inv":           0.05,
    "roe":                     0.05,
    "gross_profitability":     0.05,
    "revenue_growth":          0.05,
}

TURNAROUND_WEIGHTS = {
    "z_score_trajectory":   0.15,
    "z_score_level":        0.10,
    "beneish_m_screen":     0.10,
    "revenue_trajectory":   0.12,
    "margin_trajectory":    0.12,
    "fcf_trajectory":       0.10,
    "insider_cluster":      0.08,
    "short_interest_inv":   0.05,
    "rsi_divergence":       0.05,
    "industry_cycle":       0.08,
    "relative_value":       0.05,
}

SWING_WEIGHTS = {
    "sue_magnitude":        0.15,
    "ear_signal":           0.10,
    "momentum_6m":          0.10,
    "sector_momentum":      0.10,
    "rsi_signal":           0.15,
    "volume_confirmation":  0.08,
    "macd_signal":          0.10,
    "atr_reward_risk":      0.10,
    "obv_trend":            0.07,
    "bollinger_position":   0.05,
}

# ─── Conservative Index Factor Weights ───────────────────────────────────────
# Public-facing index: Quality + Value + Low-Vol + Income
# Target audience: Retirement/long-term investors (10+ year horizon)
# Reference factors: Novy-Marx (2013), Piotroski (2000), Altman (1968)
CONSERVATIVE_WEIGHTS = {
    # Quality (40%)
    "gross_profitability":   0.15,  # Novy-Marx (2013) — most robust quality factor
    "roic":                  0.10,  # Return on invested capital
    "piotroski_f":           0.08,  # Balance sheet quality screen
    "altman_z_norm":         0.07,  # Financial health (higher = safer)
    # Value (25%)
    "earnings_yield":        0.10,  # EBIT/EV
    "fcf_yield":             0.08,  # FCF/EV
    "ev_to_ebitda_inv":      0.07,  # Inverse EV/EBITDA (lower multiple = better)
    # Low-Volatility Safety (20%)
    "low_vol_252d_inv":      0.12,  # Inverted 252-day realized vol
    "debt_to_equity_inv":    0.05,  # Low leverage
    "current_ratio":         0.03,  # Liquidity buffer
    # Income (15%)
    "dividend_yield":        0.06,  # Current dividend yield
    "div_growth_5y":         0.05,  # 5-year dividend CAGR
    "consecutive_increases": 0.04,  # Dividend growth streak (aristocrat signal)
}

# ─── Aggressive Index Factor Weights ─────────────────────────────────────────
# Public-facing index: Momentum + Growth + Quality filter
# Target audience: Younger investors building long-horizon portfolios
# Reference: Jegadeesh & Titman (1993) momentum, Gu et al. (2020) ML predictors
AGGRESSIVE_WEIGHTS = {
    # Momentum (35%)
    "momentum_12m":          0.20,  # 12-minus-1 month (FF standard, skip 1 month)
    "momentum_6m":           0.10,  # 6-month momentum
    "sector_momentum":       0.05,  # Sector-level tailwind
    # Growth (30%)
    "revenue_cagr_5y":       0.12,  # Revenue growth trajectory
    "earnings_cagr_5y":      0.10,  # Earnings acceleration
    "revenue_trajectory":    0.08,  # Recent quarter trend
    # Quality filter — avoid "junk growth" (unprofitable growers) (25%)
    "gross_profitability":   0.10,  # Profitability screen
    "roic":                  0.08,  # Capital efficiency
    "piotroski_f":           0.07,  # Balance sheet health
    # Technical confirmation (10%)
    "rsi_signal":            0.05,  # Not overbought filter
    "volume_confirmation":   0.05,  # Institutional support
}

STRATEGY_WEIGHTS = {
    "long_term":    LONG_TERM_WEIGHTS,
    "dividend":     DIVIDEND_WEIGHTS,
    "turnaround":   TURNAROUND_WEIGHTS,
    "swing":        SWING_WEIGHTS,
    "conservative": CONSERVATIVE_WEIGHTS,
    "aggressive":   AGGRESSIVE_WEIGHTS,
}

INITIAL_WEIGHT_VERSION = "2026-Q1"

# Maximum factor counts per strategy
MAX_FACTORS = {
    "long_term":    15,
    "dividend":     15,
    "turnaround":   12,
    "swing":        12,
    "conservative": 13,
    "aggressive":   11,
}

# ─── Methodology Flags ────────────────────────────────────────────────────────
# Set to True to apply sector neutralization in scoring (removes sector bets).
# See scoring/utils.py::sector_neutralize().
SECTOR_NEUTRALIZE = True

# ─── Technical Indicator Parameters ─────────────────────────────────────────
SMA_SHORT   = 50
SMA_LONG    = 200
EMA_SHORT   = 20
RSI_PERIOD  = 14
RSI_SHORT   = 5
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9
BB_PERIOD   = 20
BB_STD      = 2.0
ATR_PERIOD  = 14
VOL_SMA     = 20
DIVERGENCE_WINDOW = 14
BREAKOUT_VOLUME_MULT = 1.5

# ─── Position Sizing ─────────────────────────────────────────────────────────
TURNAROUND_MAX_POSITION = 0.05   # 5% max per position in turnaround index
TURNAROUND_TOTAL_MAX    = 0.20   # 20% total allocation
SWING_RISK_PER_TRADE    = 0.02   # 2% risk per trade
SWING_MAX_CONCURRENT    = 10
SWING_ATR_STOP_MULT_LOW  = 1.5
SWING_ATR_STOP_MULT_HIGH = 2.0
SWING_MIN_RR_RATIO       = 2.0   # Minimum reward:risk

# ─── Altman Z-Score Thresholds ───────────────────────────────────────────────
ALTMAN_SAFE_ZONE    = 2.99
ALTMAN_GREY_ZONE    = 1.81
ALTMAN_DISTRESS     = 1.81   # Exclude if Z < 1.81

# ─── Beneish M-Score Threshold ───────────────────────────────────────────────
BENEISH_THRESHOLD = -1.78  # M > -1.78 → likely manipulation → exclude

# ─── Chowder Rule ────────────────────────────────────────────────────────────
CHOWDER_STANDARD  = 12.0  # yield + 5y_div_growth >= 12%
CHOWDER_UTILITY   = 8.0   # For utilities/REITs

# ─── HMM Regime ──────────────────────────────────────────────────────────────
HMM_N_STATES    = 3
HMM_HISTORY_YEARS = 10
HMM_REGIME_LABELS = {0: "bear", 1: "neutral", 2: "bull"}

# ─── Benchmark ───────────────────────────────────────────────────────────────
BENCHMARK_TICKER = "SPY"

# ─── Swing PEAD Protocol ─────────────────────────────────────────────────────
PEAD_ENTRY_DELAY_DAYS = 2    # Enter Day 2 after positive surprise
PEAD_MIN_HOLD_DAYS    = 21
PEAD_MAX_HOLD_DAYS    = 63   # Time stop

# ─── Paper Trading Simulation ────────────────────────────────────────────────
# Slippage applied to each paper trade (both buys and sells).
# 0.05% is conservative for liquid large-caps; increase for smaller names.
PAPER_SLIPPAGE      = 0.0005     # 0.05% per trade
PAPER_STARTING_CASH = 1000.00   # Default starting value for each simulated portfolio

# Set this to the date you first started running the live system (ISO format: "YYYY-MM-DD").
# Everything BEFORE this date is backsimulated (survivorship bias applies).
# Everything FROM this date onward is genuine out-of-sample tracking.
# Leave as None until you're actually ready to go live — it will default to today.
LIVE_START_DATE = "2026-03-15"  # Set to today — everything before this is backsimulated

# ─── Shiller CAPE Data ────────────────────────────────────────────────────────
# Robert Shiller's publicly available .xls file (Yale Economics).
# Requires xlrd>=2.0 for .xls format support.
CAPE_URL                       = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
CAPE_STRETCHED_THRESHOLD       = 30.0   # CAPE > 30 → "market_stretched" flag
CAPE_EXTREME_THRESHOLD         = 35.0   # CAPE > 35 → "market_extreme" flag

# ─── Sector Valuation Flags ───────────────────────────────────────────────────
# A sector is flagged as stretched if 2+ of (P/S, P/E, EV/EBITDA) are
# more than this multiple above their own 5-year historical median.
SECTOR_VALUATION_STRETCH_MULTIPLIER = 1.5

# ─── Market Health Score Component Weights ───────────────────────────────────
# Must sum to 1.0. Each component is scored 0-100 before weighting.
MARKET_HEALTH_WEIGHTS = {
    "regime":           0.25,   # HMM regime state
    "yield_curve":      0.20,   # 10Y-2Y spread
    "credit_spread":    0.20,   # HY OAS (BAMLH0A0HYM2)
    "vix":              0.15,   # VIX level
    "sahm_rule":        0.10,   # Sahm Rule recession indicator
    "pct_above_200sma": 0.10,   # Market breadth
}

# ─── Email & Alerts ──────────────────────────────────────────────────────────
# Credentials loaded at runtime from secrets/email_config.json
# See alerts/email_alerts.py for setup instructions.
_email_config_file = SECRETS_DIR / "email_config.json"
ALERT_EMAIL: str = ""                    # Populated from email_config.json at runtime
NEWSLETTER_RECIPIENTS: list = []         # Populated from email_config.json at runtime

# ─── Newsletter ───────────────────────────────────────────────────────────────
NEWSLETTER_ISSUE_ONE_DATE = "2026-03-22"  # Date of first newsletter (first Saturday run)

# ─── Future-Phase Placeholders ────────────────────────────────────────────────
# Phase 3: Volatility targeting (scale exposure = VOL_TARGET / realized_60d_vol)
VOL_TARGET = 0.15          # 15% annualized target portfolio volatility

# Phase 3: CVaR risk limit
CVAR_LIMIT_95 = 0.12       # 12% monthly CVaR (95th pctl) triggers exposure reduction

# ─── Disclaimer ──────────────────────────────────────────────────────────────
DISCLAIMER = (
    "⚠️ NOT FINANCIAL ADVICE. Horizon Ledger is a personal research tool. "
    "All outputs are for informational purposes only. Past performance does not "
    "guarantee future results. Consult a licensed financial advisor before making "
    "any investment decisions."
)
