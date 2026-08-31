import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class LeadLagResult:
    leader: str
    follower: str
    lag_bars: int
    lag_minutes: int
    correlation: float
    sample_size: int


def lead_lag_correlation(leader_ret: pd.Series, follower_ret: pd.Series, max_lag: int) -> pd.DataFrame:
    results = []
    for lag in range(max_lag + 1):
        if lag == 0:
            corr = leader_ret.corr(follower_ret)
        else:
            corr = leader_ret.corr(follower_ret.shift(-lag))
        results.append({'lag_bars': lag, 'correlation': corr})
    return pd.DataFrame(results)


def find_optimal_lag(leader_ret: pd.Series, follower_ret: pd.Series, max_lag: int) -> LeadLagResult:
    df = lead_lag_correlation(leader_ret, follower_ret, max_lag)
    valid = df.dropna()
    if valid.empty:
        return LeadLagResult('', '', 0, 0, 0.0, 0)

    best = valid.loc[valid['correlation'].abs().idxmax()]
    return LeadLagResult(
        leader='',
        follower='',
        lag_bars=int(best['lag_bars']),
        lag_minutes=int(best['lag_bars'] * 10),
        correlation=best['correlation'],
        sample_size=len(leader_ret) - int(best['lag_bars'])
    )


def conditional_lead_lag(
    df: pd.DataFrame,
    condition_col: str,
    condition_value,
    max_lag: int,
    leader: str = 'gold',
    follower: str = 'silver'
) -> pd.DataFrame:
    mask = df[condition_col] == condition_value
    cond_df = df[mask].copy()

    if len(cond_df) < max_lag + 10:
        return pd.DataFrame()

    leader_ret = cond_df[f'{leader}_log_return']
    follower_ret = cond_df[f'{follower}_log_return']

    return lead_lag_correlation(leader_ret, follower_ret, max_lag)


def analyze_lead_lag(
    df: pd.DataFrame,
    max_lag: int,
    session_col: str = 'session',
    volatility_col: str = 'volatility_regime'
) -> Dict:
    gold_ret = df['gold_log_return']
    silver_ret = df['silver_log_return']

    results = {}

    gold_silver = lead_lag_correlation(gold_ret, silver_ret, max_lag)
    gold_silver['direction'] = 'Gold -> Silver'
    results['gold_to_silver'] = gold_silver

    silver_gold = lead_lag_correlation(silver_ret, gold_ret, max_lag)
    silver_gold['direction'] = 'Silver -> Gold'
    results['silver_to_gold'] = silver_gold

    best_gs = find_optimal_lag(gold_ret, silver_ret, max_lag)
    best_gs.leader = 'Gold'
    best_gs.follower = 'Silver'
    results['best_gold_silver'] = best_gs

    best_sg = find_optimal_lag(silver_ret, gold_ret, max_lag)
    best_sg.leader = 'Silver'
    best_sg.follower = 'Gold'
    results['best_silver_gold'] = best_sg

    for session in df[session_col].unique():
        session_data = df[df[session_col] == session]
        if len(session_data) > max_lag + 10:
            gs = lead_lag_correlation(session_data['gold_log_return'], session_data['silver_log_return'], max_lag)
            sg = lead_lag_correlation(session_data['silver_log_return'], session_data['gold_log_return'], max_lag)
            results[f'session_{session}'] = {
                'gold_to_silver': gs,
                'silver_to_gold': sg
            }

    for regime in df[volatility_col].unique():
        regime_data = df[df[volatility_col] == regime]
        if len(regime_data) > max_lag + 10:
            gs = lead_lag_correlation(regime_data['gold_log_return'], regime_data['silver_log_return'], max_lag)
            sg = lead_lag_correlation(regime_data['silver_log_return'], regime_data['gold_log_return'], max_lag)
            results[f'regime_{regime}'] = {
                'gold_to_silver': gs,
                'silver_to_gold': sg
            }

    return results