import pandas as pd
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
from statsmodels.tsa.stattools import adfuller, coint
from scipy import stats


@dataclass
class StationarityResult:
    adf_statistic: float
    p_value: float
    is_stationary: bool
    critical_values: Dict[str, float]


@dataclass
class CointegrationResult:
    coint_statistic: float
    p_value: float
    is_cointegrated: bool
    critical_values: Dict[str, float]
    hedge_ratio: float


@dataclass
class HalfLifeResult:
    half_life: float
    mean_reversion_speed: float


def adf_test(series: pd.Series, significance: float = 0.05) -> StationarityResult:
    clean = series.dropna()
    if len(clean) < 20:
        return StationarityResult(0, 1, False, {})

    result = adfuller(clean, autolag='AIC')
    return StationarityResult(
        adf_statistic=result[0],
        p_value=result[1],
        is_stationary=result[1] < significance,
        critical_values=result[4]
    )


def engle_granger_test(
    series1: pd.Series,
    series2: pd.Series,
    significance: float = 0.05
) -> CointegrationResult:
    clean1 = series1.dropna()
    clean2 = series2.dropna()

    common_idx = clean1.index.intersection(clean2.index)
    if len(common_idx) < 20:
        return CointegrationResult(0, 1, False, {}, 0)

    s1 = clean1.loc[common_idx]
    s2 = clean2.loc[common_idx]

    result = coint(s1, s2)
    return CointegrationResult(
        coint_statistic=result[0],
        p_value=result[1],
        is_cointegrated=result[1] < significance,
        critical_values=dict(zip(['1%', '5%', '10%'], result[2])),
        hedge_ratio=0
    )


def calculate_half_life(spread: pd.Series) -> HalfLifeResult:
    clean = spread.dropna()
    if len(clean) < 20:
        return HalfLifeResult(0, 0)

    y = clean.diff().dropna()
    x = clean.shift(1).dropna()

    common_idx = y.index.intersection(x.index)
    y = y.loc[common_idx]
    x = x.loc[common_idx]

    if len(x) < 10:
        return HalfLifeResult(0, 0)

    x = np.column_stack([x, np.ones(len(x))])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]

    if beta[0] >= 0:
        return HalfLifeResult(np.inf, 0)

    half_life = -np.log(2) / beta[0]
    return HalfLifeResult(half_life, -beta[0])


def autocorrelation(series: pd.Series, max_lag: int = 20) -> pd.Series:
    return pd.Series([series.autocorr(lag=i) for i in range(max_lag + 1)], index=range(max_lag + 1))


def correlation_stability(
    series1: pd.Series,
    series2: pd.Series,
    window: int,
    step: int = None
) -> Dict:
    if step is None:
        step = window // 2

    correlations = []
    for i in range(0, len(series1) - window, step):
        corr = series1.iloc[i:i+window].corr(series2.iloc[i:i+window])
        correlations.append(corr)

    return {
        'mean': np.mean(correlations),
        'std': np.std(correlations),
        'min': np.min(correlations),
        'max': np.max(correlations),
        'count': len(correlations)
    }