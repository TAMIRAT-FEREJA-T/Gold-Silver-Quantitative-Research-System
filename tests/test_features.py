import pandas as pd
import numpy as np
import pytest
from src.features import (
    calculate_log_returns, calculate_pct_returns, calculate_ema,
    calculate_rsi, calculate_atr, calculate_adx, calculate_momentum
)


def test_calculate_log_returns():
    series = pd.Series([100, 101, 102, 101, 100], index=pd.date_range('2024-01-01', periods=5, freq='10min'))
    returns = calculate_log_returns(series)
    expected = np.log(series / series.shift(1))
    pd.testing.assert_series_equal(returns, expected)


def test_calculate_pct_returns():
    series = pd.Series([100, 101, 102, 101, 100], index=pd.date_range('2024-01-01', periods=5, freq='10min'))
    returns = calculate_pct_returns(series)
    expected = series.pct_change()
    pd.testing.assert_series_equal(returns, expected)


def test_calculate_ema():
    series = pd.Series([100, 101, 102, 103, 104], index=pd.date_range('2024-01-01', periods=5, freq='10min'))
    ema = calculate_ema(series, 3)
    assert len(ema) == 5
    assert not ema.isna().all()


def test_calculate_rsi():
    series = pd.Series([100, 101, 102, 101, 100, 99, 98, 99, 100, 101, 102, 103, 104, 105],
                       index=pd.date_range('2024-01-01', periods=14, freq='10min'))
    rsi = calculate_rsi(series, 14)
    assert len(rsi) == 14
    assert rsi.iloc[-1] >= 0 and rsi.iloc[-1] <= 100


def test_calculate_atr():
    high = pd.Series([102, 103, 104, 105, 106], index=pd.date_range('2024-01-01', periods=5, freq='10min'))
    low = pd.Series([99, 100, 101, 102, 103], index=pd.date_range('2024-01-01', periods=5, freq='10min'))
    close = pd.Series([101, 102, 103, 104, 105], index=pd.date_range('2024-01-01', periods=5, freq='10min'))
    atr = calculate_atr(high, low, close, 3)
    assert len(atr) == 5
    assert not atr.isna().all()


def test_calculate_adx():
    high = pd.Series([102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
                     index=pd.date_range('2024-01-01', periods=14, freq='10min'))
    low = pd.Series([99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
                    index=pd.date_range('2024-01-01', periods=14, freq='10min'))
    close = pd.Series([101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
                      index=pd.date_range('2024-01-01', periods=14, freq='10min'))
    adx = calculate_adx(high, low, close, 14)
    assert len(adx) == 14


def test_calculate_momentum():
    series = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
                       index=pd.date_range('2024-01-01', periods=13, freq='10min'))
    mom = calculate_momentum(series, [1, 2, 3, 6, 12])
    assert 'mom_1' in mom.columns
    assert 'mom_12' in mom.columns
    assert len(mom) == 13