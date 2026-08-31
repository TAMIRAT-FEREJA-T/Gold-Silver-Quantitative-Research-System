import pandas as pd
import numpy as np
import pytest
from src.data_cleaner import clean_ohlcv, synchronize_data


def test_clean_ohlcv_removes_nan():
    df = pd.DataFrame({
        'open': [100, 101, np.nan, 103],
        'high': [102, 103, 104, 105],
        'low': [99, 100, 101, 102],
        'close': [101, 102, 103, 104],
        'tick_volume': [1000, 1000, 1000, 1000],
        'spread': [2, 2, 2, 2],
        'real_volume': [100, 100, 100, 100]
    }, index=pd.date_range('2024-01-01', periods=4, freq='10min', tz='UTC'))

    cleaned, report = clean_ohlcv(df, "Test")
    assert len(cleaned) == 3
    assert report.nan_values == 1


def test_clean_ohlcv_removes_zero_prices():
    df = pd.DataFrame({
        'open': [100, 0, 102, 103],
        'high': [102, 103, 104, 105],
        'low': [99, 100, 101, 102],
        'close': [101, 102, 103, 104],
        'tick_volume': [1000, 1000, 1000, 1000],
        'spread': [2, 2, 2, 2],
        'real_volume': [100, 100, 100, 100]
    }, index=pd.date_range('2024-01-01', periods=4, freq='10min', tz='UTC'))

    cleaned, report = clean_ohlcv(df, "Test")
    assert len(cleaned) == 3
    assert report.zero_prices == 1


def test_clean_ohlcv_removes_invalid_ohlc():
    df = pd.DataFrame({
        'open': [100, 101, 102, 103],
        'high': [102, 100, 104, 105],  # high < open on row 1
        'low': [99, 100, 101, 102],
        'close': [101, 102, 103, 104],
        'tick_volume': [1000, 1000, 1000, 1000],
        'spread': [2, 2, 2, 2],
        'real_volume': [100, 100, 100, 100]
    }, index=pd.date_range('2024-01-01', periods=4, freq='10min', tz='UTC'))

    cleaned, report = clean_ohlcv(df, "Test")
    assert len(cleaned) == 3
    assert report.invalid_ohlc == 1


def test_clean_ohlcv_removes_duplicates():
    idx = pd.date_range('2024-01-01', periods=3, freq='10min', tz='UTC')
    idx = idx.insert(1, idx[1])  # duplicate
    df = pd.DataFrame({
        'open': [100, 101, 101, 103],
        'high': [102, 103, 103, 105],
        'low': [99, 100, 100, 102],
        'close': [101, 102, 102, 104],
        'tick_volume': [1000, 1000, 1000, 1000],
        'spread': [2, 2, 2, 2],
        'real_volume': [100, 100, 100, 100]
    }, index=idx)

    cleaned, report = clean_ohlcv(df, "Test")
    assert len(cleaned) == 3
    assert report.duplicate_timestamps == 1


def test_synchronize_data():
    idx1 = pd.date_range('2024-01-01', periods=5, freq='10min', tz='UTC')
    idx2 = pd.date_range('2024-01-01', periods=5, freq='10min', tz='UTC')

    gold = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [102, 103, 104, 105, 106],
        'low': [99, 100, 101, 102, 103],
        'close': [101, 102, 103, 104, 105],
        'tick_volume': [1000] * 5,
        'spread': [2] * 5,
        'real_volume': [100] * 5
    }, index=idx1)

    silver = pd.DataFrame({
        'open': [20, 21, 22, 23, 24],
        'high': [21, 22, 23, 24, 25],
        'low': [19, 20, 21, 22, 23],
        'close': [20.5, 21.5, 22.5, 23.5, 24.5],
        'tick_volume': [1000] * 5,
        'spread': [3] * 5,
        'real_volume': [100] * 5
    }, index=idx2)

    merged, report = synchronize_data(gold, silver)
    assert len(merged) == 5
    assert report['matched_bars'] == 5
    assert report['unmatched_gold'] == 0
    assert report['unmatched_silver'] == 0


def test_synchronize_data_partial_overlap():
    idx1 = pd.date_range('2024-01-01', periods=5, freq='10min', tz='UTC')
    idx2 = pd.date_range('2024-01-01 00:20', periods=5, freq='10min', tz='UTC')

    gold = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [102, 103, 104, 105, 106],
        'low': [99, 100, 101, 102, 103],
        'close': [101, 102, 103, 104, 105],
        'tick_volume': [1000] * 5,
        'spread': [2] * 5,
        'real_volume': [100] * 5
    }, index=idx1)

    silver = pd.DataFrame({
        'open': [20, 21, 22, 23, 24],
        'high': [21, 22, 23, 24, 25],
        'low': [19, 20, 21, 22, 23],
        'close': [20.5, 21.5, 22.5, 23.5, 24.5],
        'tick_volume': [1000] * 5,
        'spread': [3] * 5,
        'real_volume': [100] * 5
    }, index=idx2)

    merged, report = synchronize_data(gold, silver)
    assert len(merged) == 3  # only 3 overlapping bars
    assert report['matched_bars'] == 3
    assert report['unmatched_gold'] == 2
    assert report['unmatched_silver'] == 2