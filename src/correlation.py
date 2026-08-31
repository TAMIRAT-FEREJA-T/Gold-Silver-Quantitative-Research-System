import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from scipy import stats


@dataclass
class CorrelationStats:
    mean: float
    median: float
    std: float
    min: float
    max: float
    pct_positive: float
    pct_above_05: float
    pct_above_07: float


def rolling_correlation(series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
    return series1.rolling(window=window).corr(series2)


def rolling_spearman(series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
    def spearman(x, y):
        return stats.spearmanr(x, y)[0]
    return series1.rolling(window=window).apply(
        lambda x: spearman(x, series2.loc[x.index]),
        raw=False
    )


def correlation_statistics(corr_series: pd.Series) -> CorrelationStats:
    clean = corr_series.dropna()
    if len(clean) == 0:
        return CorrelationStats(0, 0, 0, 0, 0, 0, 0, 0)

    return CorrelationStats(
        mean=clean.mean(),
        median=clean.median(),
        std=clean.std(),
        min=clean.min(),
        max=clean.max(),
        pct_positive=(clean > 0).mean() * 100,
        pct_above_05=(clean > 0.5).mean() * 100,
        pct_above_07=(clean > 0.7).mean() * 100
    )


def calculate_rolling_correlations(df: pd.DataFrame, windows: List[int]) -> Dict[int, pd.Series]:
    gold_ret = df['gold_log_return']
    silver_ret = df['silver_log_return']

    results = {}
    for w in windows:
        results[w] = rolling_correlation(gold_ret, silver_ret, w)
    return results


def correlation_by_session(df: pd.DataFrame, session_col: str, windows: List[int]) -> Dict[str, Dict[int, float]]:
    sessions = df[session_col].unique()
    results = {}

    for session in sessions:
        session_data = df[df[session_col] == session]
        results[session] = {}
        for w in windows:
            if len(session_data) >= w:
                corr = session_data['gold_log_return'].corr(session_data['silver_log_return'])
                results[session][w] = corr
            else:
                results[session][w] = np.nan
    return results


def correlation_by_regime(df: pd.DataFrame, regime_col: str, windows: List[int]) -> Dict[str, Dict[int, float]]:
    regimes = df[regime_col].unique()
    results = {}

    for regime in regimes:
        regime_data = df[df[regime_col] == regime]
        results[regime] = {}
        for w in windows:
            if len(regime_data) >= w:
                corr = regime_data['gold_log_return'].corr(regime_data['silver_log_return'])
                results[regime][w] = corr
            else:
                results[regime][w] = np.nan
    return results