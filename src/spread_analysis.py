import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class SpreadStats:
    mean: float
    median: float
    min: float
    max: float
    pct_95: float
    pct_99: float


def calculate_spread_stats(spread: pd.Series) -> SpreadStats:
    clean = spread.dropna()
    if len(clean) == 0:
        return SpreadStats(0, 0, 0, 0, 0, 0)

    return SpreadStats(
        mean=clean.mean(),
        median=clean.median(),
        min=clean.min(),
        max=clean.max(),
        pct_95=clean.quantile(0.95),
        pct_99=clean.quantile(0.99)
    )


def spread_by_session(df: pd.DataFrame, session_col: str) -> Dict[str, SpreadStats]:
    results = {}
    for session in df[session_col].unique():
        session_data = df[df[session_col] == session]
        results[session] = calculate_spread_stats(session_data['gold_spread'])
        results[f'{session}_silver'] = calculate_spread_stats(session_data['silver_spread'])
    return results


def spread_by_hour(df: pd.DataFrame) -> Dict[int, SpreadStats]:
    results = {}
    df = df.copy()
    df['hour'] = df.index.hour
    for hour in range(24):
        hour_data = df[df['hour'] == hour]
        if len(hour_data) > 0:
            results[hour] = calculate_spread_stats(hour_data['gold_spread'])
    return results


def spread_by_volatility_regime(df: pd.DataFrame, regime_col: str) -> Dict[str, SpreadStats]:
    results = {}
    for regime in df[regime_col].unique():
        regime_data = df[df[regime_col] == regime]
        results[regime] = calculate_spread_stats(regime_data['gold_spread'])
        results[f'{regime}_silver'] = calculate_spread_stats(regime_data['silver_spread'])
    return results


def spread_cost_analysis(
    df: pd.DataFrame,
    gold_spread_col: str = 'gold_spread',
    silver_spread_col: str = 'silver_spread'
) -> Dict:
    gold_spread = df[gold_spread_col].dropna()
    silver_spread = df[silver_spread_col].dropna()

    return {
        'gold': {
            'avg_spread': gold_spread.mean(),
            'median_spread': gold_spread.median(),
            'spread_pct_of_price': (gold_spread / df.loc[gold_spread.index, 'gold_close'] * 100).mean()
        },
        'silver': {
            'avg_spread': silver_spread.mean(),
            'median_spread': silver_spread.median(),
            'spread_pct_of_price': (silver_spread / df.loc[silver_spread.index, 'silver_close'] * 100).mean()
        }
    }