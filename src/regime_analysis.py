import pandas as pd
import numpy as np
from typing import Dict


def classify_sessions(df: pd.DataFrame) -> pd.Series:
    hour = df.index.hour
    minute = df.index.minute

    session = pd.Series('OTHER', index=df.index)

    london_start = 7
    london_end = 16
    ny_start = 12
    ny_end = 21
    overlap_start = 12
    overlap_end = 16

    london_mask = (hour >= london_start) & (hour < london_end)
    ny_mask = (hour >= ny_start) & (hour < ny_end)
    overlap_mask = (hour >= overlap_start) & (hour < overlap_end)

    session[overlap_mask] = 'LONDON_NEW_YORK_OVERLAP'
    session[london_mask & ~overlap_mask] = 'LONDON'
    session[ny_mask & ~overlap_mask] = 'NEW_YORK'
    asia_mask = (hour >= 0) & (hour < 7)
    session[asia_mask] = 'ASIA'

    return session


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result['session'] = classify_sessions(result)

    for session in ['ASIA', 'LONDON', 'LONDON_NEW_YORK_OVERLAP', 'NEW_YORK']:
        result[f'is_{session.lower()}'] = (result['session'] == session).astype(int)

    return result


def regime_analysis(
    df: pd.DataFrame,
    session_col: str = 'session',
    volatility_col: str = 'volatility_regime'
) -> Dict:
    results = {}

    for session in df[session_col].unique():
        session_data = df[df[session_col] == session]
        if len(session_data) < 10:
            continue

        results[session] = {
            'observations': len(session_data),
            'correlation': session_data['gold_log_return'].corr(session_data['silver_log_return']),
            'gold_avg_return': session_data['gold_log_return'].mean(),
            'silver_avg_return': session_data['silver_log_return'].mean(),
            'gold_volatility': session_data['gold_log_return'].std(),
            'silver_volatility': session_data['silver_log_return'].std(),
            'gold_avg_spread': session_data['gold_spread'].mean(),
            'silver_avg_spread': session_data['silver_spread'].mean(),
        }

    for regime in df[volatility_col].unique():
        regime_data = df[df[volatility_col] == regime]
        if len(regime_data) < 10:
            continue

        results[f'vol_{regime}'] = {
            'observations': len(regime_data),
            'correlation': regime_data['gold_log_return'].corr(regime_data['silver_log_return']),
            'gold_avg_return': regime_data['gold_log_return'].mean(),
            'silver_avg_return': regime_data['silver_log_return'].mean(),
            'gold_volatility': regime_data['gold_log_return'].std(),
            'silver_volatility': regime_data['silver_log_return'].std(),
            'gold_avg_spread': regime_data['gold_spread'].mean(),
            'silver_avg_spread': regime_data['silver_spread'].mean(),
        }

    return results