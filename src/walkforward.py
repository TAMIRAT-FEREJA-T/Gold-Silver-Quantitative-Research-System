import pandas as pd
import numpy as np
from typing import Dict, List, Callable
from dataclasses import dataclass
from src.backtest import BacktestResult


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def create_walkforward_windows(
    df: pd.DataFrame,
    train_pct: float = 0.6,
    val_pct: float = 0.2,
    step_pct: float = 0.1,
    min_train_bars: int = 1000
) -> List[WalkForwardWindow]:
    n = len(df)
    train_size = int(n * train_pct)
    val_size = int(n * val_pct)
    step_size = int(n * step_pct)

    if train_size < min_train_bars:
        train_size = min_train_bars
        val_size = int(train_size * val_pct / train_pct)
        step_size = int(train_size * step_pct / train_pct)

    windows = []
    start = 0

    while start + train_size + val_size + step_size <= n:
        train_end = start + train_size
        val_end = train_end + val_size
        test_end = min(val_end + step_size, n)

        windows.append(WalkForwardWindow(
            train_start=df.index[start],
            train_end=df.index[train_end - 1],
            val_start=df.index[train_end],
            val_end=df.index[val_end - 1],
            test_start=df.index[val_end],
            test_end=df.index[test_end - 1]
        ))

        start += step_size

    return windows


def run_walkforward_analysis(
    df: pd.DataFrame,
    strategy_fn: Callable,
    strategy_params: Dict,
    windows: List[WalkForwardWindow],
    param_grid: Dict = None
) -> Dict:
    results = {
        'windows': [],
        'train_performance': [],
        'val_performance': [],
        'test_performance': [],
        'best_params_per_window': []
    }

    for i, window in enumerate(windows):
        train_df = df.loc[window.train_start:window.train_end]
        val_df = df.loc[window.val_start:window.val_end]
        test_df = df.loc[window.test_start:window.test_end]

        if len(train_df) < 100 or len(val_df) < 50 or len(test_df) < 50:
            continue

        best_params = strategy_params.copy()
        best_val_score = -np.inf

        if param_grid:
            for param_combo in _generate_param_combos(param_grid):
                test_params = {**strategy_params, **param_combo}
                val_result = strategy_fn(val_df, **test_params)
                score = val_result.net_return / max(abs(val_result.max_drawdown), 1e-6)
                if score > best_val_score:
                    best_val_score = score
                    best_params = test_params

        train_result = strategy_fn(train_df, **best_params)
        val_result = strategy_fn(val_df, **best_params)
        test_result = strategy_fn(test_df, **best_params)

        results['windows'].append({
            'window': i,
            'train_period': f"{window.train_start} to {window.train_end}",
            'val_period': f"{window.val_start} to {window.val_end}",
            'test_period': f"{window.test_start} to {window.test_end}"
        })
        results['train_performance'].append(_extract_metrics(train_result))
        results['val_performance'].append(_extract_metrics(val_result))
        results['test_performance'].append(_extract_metrics(test_result))
        results['best_params_per_window'].append(best_params)

    return results


def _generate_param_combos(param_grid: Dict) -> List[Dict]:
    import itertools
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _extract_metrics(result: BacktestResult) -> Dict:
    return {
        'total_trades': result.total_trades,
        'win_rate': result.win_rate,
        'avg_return': result.avg_return,
        'profit_factor': result.profit_factor,
        'expectancy': result.expectancy,
        'max_drawdown': result.max_drawdown,
        'sharpe_ratio': result.sharpe_ratio,
        'net_return': result.net_return,
        'cost_to_edge_ratio': result.cost_to_edge_ratio
    }


def summarize_walkforward(results: Dict) -> Dict:
    test_perf = results['test_performance']
    if not test_perf:
        return {}

    return {
        'num_windows': len(test_perf),
        'avg_net_return': np.mean([p['net_return'] for p in test_perf]),
        'std_net_return': np.std([p['net_return'] for p in test_perf]),
        'avg_sharpe': np.mean([p['sharpe_ratio'] for p in test_perf]),
        'avg_profit_factor': np.mean([p['profit_factor'] for p in test_perf]),
        'avg_win_rate': np.mean([p['win_rate'] for p in test_perf]),
        'avg_max_dd': np.mean([p['max_drawdown'] for p in test_perf]),
        'positive_windows': sum(1 for p in test_perf if p['net_return'] > 0),
        'consistency': sum(1 for p in test_perf if p['net_return'] > 0) / len(test_perf) * 100
    }