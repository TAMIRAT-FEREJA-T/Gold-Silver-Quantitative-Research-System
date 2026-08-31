import pandas as pd
import numpy as np
import pytest
from src.lead_lag import lead_lag_correlation, find_optimal_lag


def test_lead_lag_correlation():
    np.random.seed(42)
    gold_ret = pd.Series(np.random.randn(100) * 0.001)
    silver_ret = gold_ret.shift(2) * 0.8 + np.random.randn(100) * 0.0005

    result = lead_lag_correlation(gold_ret, silver_ret, 5)
    assert len(result) == 6  # 0 to 5
    assert 'lag_bars' in result.columns
    assert 'correlation' in result.columns

    # Max correlation should be around lag 2
    best_lag = result.loc[result['correlation'].abs().idxmax(), 'lag_bars']
    assert best_lag == 2


def test_lead_lag_correlation_no_lead():
    np.random.seed(42)
    gold_ret = pd.Series(np.random.randn(100) * 0.001)
    silver_ret = pd.Series(np.random.randn(100) * 0.001)

    result = lead_lag_correlation(gold_ret, silver_ret, 5)
    correlations = result['correlation'].dropna().abs()
    # Random series should have low correlations
    assert correlations.max() < 0.3


def test_find_optimal_lag():
    np.random.seed(42)
    gold_ret = pd.Series(np.random.randn(100) * 0.001)
    silver_ret = gold_ret.shift(3) * 0.7 + np.random.randn(100) * 0.0005

    result = find_optimal_lag(gold_ret, silver_ret, 10)
    assert result.lag_bars == 3
    assert result.correlation > 0.5


def test_find_optimal_lag_empty():
    gold_ret = pd.Series([np.nan] * 10)
    silver_ret = pd.Series([np.nan] * 10)
    result = find_optimal_lag(gold_ret, silver_ret, 5)
    assert result.correlation == 0.0