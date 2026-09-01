import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class DivergenceEvent:
    timestamp: pd.Timestamp
    zscore: float
    direction: str
    gold_return: float
    silver_return: float


def detect_divergences(
    df: pd.DataFrame,
    zscore_col: str,
    thresholds: List[float],
    cooldown_bars: int = 0
) -> List[DivergenceEvent]:
    events = []
    last_event_idx = -cooldown_bars - 1

    for threshold in thresholds:
        upper = df[df[zscore_col] >= threshold]
        lower = df[df[zscore_col] <= -threshold]

        for idx in upper.index:
            loc = df.index.get_loc(idx)
            if cooldown_bars > 0 and loc - last_event_idx <= cooldown_bars:
                continue
            events.append(DivergenceEvent(
                timestamp=idx,
                zscore=df.loc[idx, zscore_col],
                direction='SILVER_STRONG',
                gold_return=df.loc[idx, 'gold_log_return'],
                silver_return=df.loc[idx, 'silver_log_return']
            ))
            last_event_idx = loc

        for idx in lower.index:
            loc = df.index.get_loc(idx)
            if cooldown_bars > 0 and loc - last_event_idx <= cooldown_bars:
                continue
            events.append(DivergenceEvent(
                timestamp=idx,
                zscore=df.loc[idx, zscore_col],
                direction='GOLD_STRONG',
                gold_return=df.loc[idx, 'gold_log_return'],
                silver_return=df.loc[idx, 'silver_log_return']
            ))
            last_event_idx = loc

    return events


def analyze_divergence_forward_returns(
    df: pd.DataFrame,
    events: List[DivergenceEvent],
    forward_periods: List[int]
) -> Dict:
    """Analyze forward returns for all events (combined thresholds)."""
    results = {}

    for period in forward_periods:
        gold_returns = []
        silver_returns = []

        for event in events:
            if event.timestamp in df.index:
                idx = df.index.get_loc(event.timestamp)
                if idx + period < len(df):
                    gold_ret = df['gold_log_return'].iloc[idx + period]
                    silver_ret = df['silver_log_return'].iloc[idx + period]
                    gold_returns.append(gold_ret)
                    silver_returns.append(silver_ret)

        if gold_returns:
            results[f'{period}_bars'] = {
                'gold': {
                    'mean': np.mean(gold_returns),
                    'median': np.median(gold_returns),
                    'std': np.std(gold_returns),
                    'pct_positive': np.mean(np.array(gold_returns) > 0) * 100,
                    'max_favorable': np.max(gold_returns),
                    'max_adverse': np.min(gold_returns),
                    'count': len(gold_returns)
                },
                'silver': {
                    'mean': np.mean(silver_returns),
                    'median': np.median(silver_returns),
                    'std': np.std(silver_returns),
                    'pct_positive': np.mean(np.array(silver_returns) > 0) * 100,
                    'max_favorable': np.max(silver_returns),
                    'max_adverse': np.min(silver_returns),
                    'count': len(silver_returns)
                }
            }

    return results


def analyze_divergence_by_threshold(
    df: pd.DataFrame,
    zscore_col: str,
    thresholds: List[float],
    forward_periods: List[int],
    cooldown_bars: int = 0
) -> Dict:
    """Analyze forward returns for each threshold separately."""
    results = {}

    for threshold in thresholds:
        events = detect_divergences(df, zscore_col, [threshold], cooldown_bars)
        results[threshold] = analyze_divergence_forward_returns(df, events, forward_periods)

    return results


def analyze_divergence_by_direction(
    df: pd.DataFrame,
    events: List[DivergenceEvent],
    forward_periods: List[int]
) -> Dict:
    silver_strong = [e for e in events if e.direction == 'SILVER_STRONG']
    gold_strong = [e for e in events if e.direction == 'GOLD_STRONG']

    return {
        'silver_strong': analyze_divergence_forward_returns(df, silver_strong, forward_periods),
        'gold_strong': analyze_divergence_forward_returns(df, gold_strong, forward_periods)
    }


def conditional_divergence_analysis(
    df: pd.DataFrame,
    zscore_col: str,
    condition_col: str,
    condition_values: List,
    thresholds: List[float],
    forward_periods: List[int]
) -> Dict:
    results = {}

    for cond_val in condition_values:
        cond_df = df[df[condition_col] == cond_val]
        if len(cond_df) < 10:
            continue

        events = detect_divergences(cond_df, zscore_col, thresholds)
        results[cond_val] = analyze_divergence_forward_returns(cond_df, events, forward_periods)

    return results