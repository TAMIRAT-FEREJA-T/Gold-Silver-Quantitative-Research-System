import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    total_bars: int
    missing_timestamps: int
    nan_values: int
    zero_prices: int
    invalid_ohlc: int
    duplicate_timestamps: int
    gaps: list


def clean_ohlcv(df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, DataQualityReport]:
    original_len = len(df)

    issues = []

    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        issues.append(f"NaN values: {nan_count}")
        df = df.dropna()

    zero_price = ((df['open'] == 0) | (df['high'] == 0) | (df['low'] == 0) | (df['close'] == 0)).sum()
    if zero_price > 0:
        issues.append(f"Zero prices: {zero_price}")
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]

    invalid_ohlc = ((df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close']) |
                    (df['low'] > df['open']) | (df['low'] > df['close']) | (df['open'] > df['high']) |
                    (df['open'] < df['low']) | (df['close'] > df['high']) | (df['close'] < df['low'])).sum()
    if invalid_ohlc > 0:
        issues.append(f"Invalid OHLC: {invalid_ohlc}")
        df = df[~((df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close']) |
                  (df['low'] > df['open']) | (df['low'] > df['close']) | (df['open'] > df['high']) |
                  (df['open'] < df['low']) | (df['close'] > df['high']) | (df['close'] < df['low']))]

    dup_count = df.index.duplicated().sum()
    if dup_count > 0:
        issues.append(f"Duplicate timestamps: {dup_count}")
        df = df[~df.index.duplicated(keep='first')]

    df = df.sort_index()

    expected_freq = pd.infer_freq(df.index[:100])
    missing = []
    if expected_freq:
        expected_index = pd.date_range(df.index[0], df.index[-1], freq=expected_freq, tz='UTC')
        missing_ts = expected_index.difference(df.index)
        if len(missing_ts) > 0:
            issues.append(f"Missing timestamps: {len(missing_ts)}")
            missing = missing_ts.tolist()[:10]

    if issues:
        logger.warning(f"{symbol} data quality issues: {'; '.join(issues)}")

    report = DataQualityReport(
        total_bars=original_len,
        missing_timestamps=len(missing),
        nan_values=nan_count,
        zero_prices=zero_price,
        invalid_ohlc=invalid_ohlc,
        duplicate_timestamps=dup_count,
        gaps=missing
    )

    return df, report


def synchronize_data(gold_df: pd.DataFrame, silver_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    gold_cols = {c: f'gold_{c}' for c in gold_df.columns}
    silver_cols = {c: f'silver_{c}' for c in silver_df.columns}

    gold_df = gold_df.rename(columns=gold_cols)
    silver_df = silver_df.rename(columns=silver_cols)

    merged = pd.merge(gold_df, silver_df, left_index=True, right_index=True, how='inner')
    merged.sort_index(inplace=True)

    gold_only = gold_df.index.difference(silver_df.index)
    silver_only = silver_df.index.difference(gold_df.index)

    sync_report = {
        'gold_bars': len(gold_df),
        'silver_bars': len(silver_df),
        'matched_bars': len(merged),
        'unmatched_gold': len(gold_only),
        'unmatched_silver': len(silver_only),
        'missing_periods': []
    }

    if len(gold_only) > 0:
        logger.warning(f"Gold has {len(gold_only)} unmatched timestamps")
    if len(silver_only) > 0:
        logger.warning(f"Silver has {len(silver_only)} unmatched timestamps")

    expected_freq = pd.infer_freq(merged.index[:100])
    if expected_freq:
        expected_index = pd.date_range(merged.index[0], merged.index[-1], freq=expected_freq, tz='UTC')
        missing = expected_index.difference(merged.index)
        sync_report['missing_periods'] = missing.tolist()[:10]
        if len(missing) > 0:
            logger.warning(f"Synchronized data has {len(missing)} missing periods")

    return merged, sync_report