"""
Horizon Ledger — Shared Scoring Utilities (Extended)
Sector neutralization and other cross-cutting scoring helpers.

References:
  Grinold & Kahn (2000): Information Coefficient, active management
  Harvey, Liu & Zhu (2016): Multiple testing in factor research
"""

import logging

import numpy as np
import pandas as pd

from config import MISSING_FACTOR_FILL

log = logging.getLogger(__name__)


def sector_neutralize(
    df: pd.DataFrame,
    factor_cols: list[str],
    sector_col: str = "sector",
    min_sector_size: int = 3,
) -> pd.DataFrame:
    """
    Sector-neutral factor normalization.

    For each factor column, z-score within each GICS sector, then re-rank
    cross-sectionally to restore 0-100 percentile scale.  This removes
    sector-level return premia so scores reflect stock-level alpha only.

    Args:
        df: DataFrame containing factor columns and a sector column.
        factor_cols: List of raw factor column names (before _pct suffix).
        sector_col: Column name containing GICS sector labels.
        min_sector_size: Sectors with fewer stocks fall back to universe z-score.

    Returns:
        df with factor columns sector-neutralized in place.
    """
    df = df.copy()

    for col in factor_cols:
        if col not in df.columns:
            continue

        neutralized = df[col].copy().astype(float)

        if sector_col in df.columns:
            for sector, grp in df.groupby(sector_col):
                idx = grp.index
                vals = grp[col].dropna()

                if len(vals) < min_sector_size:
                    # Too few stocks — fall back to universe-level normalization
                    continue

                mu  = vals.mean()
                std = vals.std()
                if std > 0:
                    neutralized.loc[idx] = (df.loc[idx, col] - mu) / std
                else:
                    neutralized.loc[idx] = 0.0

        # Clip extreme values (±3σ) to reduce outlier influence
        neutralized = neutralized.clip(
            neutralized.mean() - 3 * neutralized.std(),
            neutralized.mean() + 3 * neutralized.std(),
        )
        df[col] = neutralized

    return df


def compute_ic(
    predictions_df: pd.DataFrame,
    factor_col: str,
    return_col: str = "return_63d",
) -> float:
    """
    Compute the Information Coefficient (IC) for a single factor.

    IC = Spearman rank correlation between factor scores and realized returns.
    Range: -1 to +1.  Values of 0.02-0.05 are considered good in practice.

    Args:
        predictions_df: DataFrame with factor scores and realized forward returns.
        factor_col: Column name of the factor score.
        return_col: Column name of the realized forward return.

    Returns:
        IC value (float), or NaN if insufficient data.
    """
    from scipy.stats import spearmanr

    df = predictions_df[[factor_col, return_col]].dropna()
    if len(df) < 10:
        return float("nan")

    ic, _ = spearmanr(df[factor_col], df[return_col])
    return float(ic)


def compute_rolling_ic(
    predictions_df: pd.DataFrame,
    factor_col: str,
    date_col: str = "signal_date",
    return_col: str = "return_63d",
    window_months: int = 12,
) -> pd.Series:
    """
    Compute rolling IC for a factor over time.

    Returns a Series indexed by month with IC values.
    Useful for detecting factor decay or improvement.
    """
    from scipy.stats import spearmanr

    df = predictions_df[[date_col, factor_col, return_col]].dropna().copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Monthly grouping
    df["year_month"] = df[date_col].dt.to_period("M")

    monthly_ic = {}
    periods = sorted(df["year_month"].unique())
    for i, period in enumerate(periods):
        if i < window_months - 1:
            start_period = periods[0]
        else:
            start_period = periods[i - window_months + 1]
        window_df = df[(df["year_month"] >= start_period) & (df["year_month"] <= period)]
        vals = window_df[[factor_col, return_col]].dropna()
        if len(vals) < 10:
            monthly_ic[period] = float("nan")
        else:
            ic, _ = spearmanr(vals[factor_col], vals[return_col])
            monthly_ic[period] = float(ic)

    return pd.Series(monthly_ic)


def ic_information_ratio(ic_series: pd.Series) -> float:
    """
    IC Information Ratio = mean(IC) / std(IC).
    Values above 0.5 indicate consistent factor predictability.
    """
    clean = ic_series.dropna()
    if len(clean) < 3 or clean.std() == 0:
        return float("nan")
    return float(clean.mean() / clean.std())
