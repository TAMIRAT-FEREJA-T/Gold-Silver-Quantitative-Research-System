import pandas as pd
import numpy as np
from typing import Optional


def calculate_log_returns(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1))


def calculate_pct_returns(series: pd.Series) -> pd.Series:
    return series.pct_change()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    return adx


def calculate_momentum(series: pd.Series, periods: list[int]) -> pd.DataFrame:
    df = pd.DataFrame(index=series.index)
    for p in periods:
        df[f'mom_{p}'] = series.pct_change(p)
    return df


def add_features(df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    result = df.copy()

    close_col = f'{prefix}close' if prefix else 'close'
    high_col = f'{prefix}high' if prefix else 'high'
    low_col = f'{prefix}low' if prefix else 'low'

    if close_col not in result.columns:
        return result

    result[f'{prefix}log_return'] = calculate_log_returns(result[close_col])
    result[f'{prefix}pct_return'] = calculate_pct_returns(result[close_col])

    for period in [9, 21, 50, 200]:
        result[f'{prefix}ema_{period}'] = calculate_ema(result[close_col], period)

    result[f'{prefix}rsi_14'] = calculate_rsi(result[close_col], 14)
    result[f'{prefix}atr_14'] = calculate_atr(result[high_col], result[low_col], result[close_col], 14)
    result[f'{prefix}adx_14'] = calculate_adx(result[high_col], result[low_col], result[close_col], 14)

    momentum_periods = [1, 2, 3, 6, 12]
    mom_df = calculate_momentum(result[close_col], momentum_periods)
    for col in mom_df.columns:
        result[f'{prefix}{col}'] = mom_df[col]

    return result


def calculate_volatility_regime(returns: pd.Series, window: int = 100) -> pd.Series:
    vol = returns.rolling(window=window).std()
    percentile = vol.rolling(window=window*5).rank(pct=True)

    regime = pd.Series('NORMAL', index=returns.index)
    regime[percentile <= 0.2] = 'LOW'
    regime[percentile >= 0.8] = 'HIGH'
    return regime


def calculate_atr_pct(atr: pd.Series, close: pd.Series) -> pd.Series:
    return atr / close * 100