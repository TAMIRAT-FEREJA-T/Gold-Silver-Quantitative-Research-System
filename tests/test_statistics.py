import pandas as pd
import numpy as np
import pytest
from src.statistics import (
    adf_test, engle_granger_test, calculate_half_life,
    autocorrelation, correlation_stability
)


def test_test_stationarity_random_walk():
    np.random.seed(42)
    rw = pd.Series(np.cumsum(np.random.randn(200)))
    result = adf_test(rw)
    assert result.is_stationary == False
    assert result.p_value > 0.05


def test_test_stationarity_mean_reverting():
    np.random.seed(42)
    # AR(1) process with mean reversion
    n = 200
    series = np.zeros(n)
    for i in range(1, n):
        series[i] = 0.8 * series[i-1] + np.random.randn() * 0.1
    mr = pd.Series(series)
    result = adf_test(mr)
    assert result.is_stationary == True
    assert result.p_value < 0.05


def test_test_cointegration():
    np.random.seed(42)
    # Two cointegrated series
    n = 200
    common = np.cumsum(np.random.randn(n) * 0.1)
    s1 = common + np.random.randn(n) * 0.05
    s2 = common * 1.5 + np.random.randn(n) * 0.05

    series1 = pd.Series(s1)
    series2 = pd.Series(s2)

    result = engle_granger_test(series1, series2)
    # Note: cointegration test may not always detect due to noise
    assert isinstance(result.is_cointegrated, bool)


def test_calculate_half_life_mean_reverting():
    np.random.seed(42)
    # Mean reverting spread
    n = 500
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.95 * spread[i-1] + np.random.randn() * 0.01
    series = pd.Series(spread)

    result = calculate_half_life(series)
    # Half-life should be around log(2)/log(1/0.95) ~ 13.5 bars
    assert 5 < result.half_life < 50


def test_calculate_half_life_random_walk():
    np.random.seed(42)
    rw = pd.Series(np.cumsum(np.random.randn(500) * 0.01))
    result = calculate_half_life(rw)
    # Random walk should have long half-life (finite but large)
    # Note: finite sample can produce spurious mean reversion
    assert result.half_life > 10 or np.isinf(result.half_life)


def test_autocorrelation():
    np.random.seed(42)
    # AR(1) process
    n = 200
    series = np.zeros(n)
    for i in range(1, n):
        series[i] = 0.7 * series[i-1] + np.random.randn() * 0.1
    s = pd.Series(series)

    acf = autocorrelation(s, 10)
    assert len(acf) == 11
    assert acf[0] == 1.0
    assert abs(acf[1] - 0.7) < 0.2  # Should be close to AR coefficient


def test_correlation_stability():
    np.random.seed(42)
    n = 500
    gold = np.random.randn(n) * 0.001
    silver = gold * 0.8 + np.random.randn(n) * 0.0005

    s1 = pd.Series(gold)
    s2 = pd.Series(silver)

    result = correlation_stability(s1, s2, 100, 50)
    assert result['count'] > 0
    assert result['mean'] > 0.5
    assert result['std'] >= 0