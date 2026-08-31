import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class RatioStats:
    current: float
    mean: float
    std: float
    percentile: float
    zscore: float


def calculate_ratio(gold_close: pd.Series, silver_close: pd.Series) -> pd.Series:
    return gold_close / silver_close


def calculate_ratio_features(
    ratio: pd.Series,
    windows: List[int],
    zscore_window: int
) -> pd.DataFrame:
    df = pd.DataFrame(index=ratio.index)
    df['ratio'] = ratio

    for w in windows:
        df[f'ratio_mean_{w}'] = ratio.rolling(window=w).mean()
        df[f'ratio_std_{w}'] = ratio.rolling(window=w).std()
        df[f'ratio_percentile_{w}'] = ratio.rolling(window=w).apply(
            lambda x: (x <= x.iloc[-1]).mean() * 100 if len(x) > 0 else np.nan,
            raw=False
        )

    df[f'ratio_zscore_{zscore_window}'] = (
        ratio - ratio.rolling(window=zscore_window).mean()
    ) / ratio.rolling(window=zscore_window).std()

    return df


def analyze_ratio_extremes(
    df: pd.DataFrame,
    zscore_col: str,
    thresholds: List[float],
    forward_periods: List[int]
) -> Dict:
    results = {}

    for threshold in thresholds:
        upper_events = df[df[zscore_col] >= threshold]
        lower_events = df[df[zscore_col] <= -threshold]

        results[threshold] = {
            'upper': analyze_events(df, upper_events.index, forward_periods, 'ratio'),
            'lower': analyze_events(df, lower_events.index, forward_periods, 'ratio')
        }

    return results


def analyze_events(
    df: pd.DataFrame,
    event_times: pd.DatetimeIndex,
    forward_periods: List[int],
    target_col: str
) -> Dict:
    if len(event_times) == 0:
        return {}

    returns = {}
    for period in forward_periods:
        fwd_returns = []
        for t in event_times:
            if t in df.index:
                idx = df.index.get_loc(t)
                if idx + period < len(df):
                    current = df[target_col].iloc[idx]
                    future = df[target_col].iloc[idx + period]
                    ret = (future - current) / current
                    fwd_returns.append(ret)

        if fwd_returns:
            returns[f'{period}_bars'] = {
                'mean': np.mean(fwd_returns),
                'median': np.median(fwd_returns),
                'std': np.std(fwd_returns),
                'pct_positive': np.mean(np.array(fwd_returns) > 0) * 100,
                'max_favorable': np.max(fwd_returns),
                'max_adverse': np.min(fwd_returns),
                'count': len(fwd_returns)
            }

    return returns


def ratio_reversion_test(
    df: pd.DataFrame,
    zscore_col: str,
    thresholds: List[float],
    reversion_levels: List[float]
) -> Dict:
    results = {}

    for threshold in thresholds:
        upper_events = df[df[zscore_col] >= threshold]
        lower_events = df[df[zscore_col] <= -threshold]

        for level in reversion_levels:
            upper_revert = 0
            lower_revert = 0
            upper_times = []
            lower_times = []

            for t in upper_events.index:
                if t in df.index:
                    idx = df.index.get_loc(t)
                    for i in range(idx + 1, min(idx + 200, len(df))):
                        if abs(df[zscore_col].iloc[i]) <= level:
                            upper_revert += 1
                            upper_times.append(i - idx)
                            break

            for t in lower_events.index:
                if t in df.index:
                    idx = df.index.get_loc(t)
                    for i in range(idx + 1, min(idx + 200, len(df))):
                        if abs(df[zscore_col].iloc[i]) <= level:
                            lower_revert += 1
                            lower_times.append(i - idx)
                            break

            results[f'z{threshold}_revert_{level}'] = {
                'upper_revert_pct': upper_revert / len(upper_events) * 100 if len(upper_events) > 0 else 0,
                'lower_revert_pct': lower_revert / len(lower_events) * 100 if len(lower_events) > 0 else 0,
                'upper_avg_time': np.mean(upper_times) if upper_times else 0,
                'lower_avg_time': np.mean(lower_times) if lower_times else 0,
                'upper_max_time': np.max(upper_times) if upper_times else 0,
                'lower_max_time': np.max(lower_times) if lower_times else 0
            }

    return results