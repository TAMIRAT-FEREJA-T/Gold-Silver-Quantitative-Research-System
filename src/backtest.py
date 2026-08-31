import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class PositionType(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position_type: PositionType
    size: float
    gross_pnl: float
    net_pnl: float
    holding_bars: int


@dataclass
class BacktestResult:
    trades: List[Trade]
    total_trades: int
    win_rate: float
    avg_return: float
    median_return: float
    avg_winner: float
    avg_loser: float
    profit_factor: float
    expectancy: float
    max_drawdown: float
    max_consecutive_losses: int
    sharpe_ratio: float
    sortino_ratio: float
    avg_holding_bars: float
    best_trade: float
    worst_trade: float
    gross_return: float
    total_costs: float
    net_return: float
    trades_per_day: float
    cost_to_edge_ratio: float


def calculate_transaction_cost(
    entry_price: float,
    exit_price: float,
    spread_entry: float,
    spread_exit: float,
    commission_per_lot: float = 0,
    slippage: float = 0
) -> float:
    spread_cost = (spread_entry + spread_exit) / 2
    total_cost = spread_cost + commission_per_lot + slippage
    return total_cost


def simulate_lead_lag_strategy(
    df: pd.DataFrame,
    leader: str,
    follower: str,
    lag_bars: int,
    entry_threshold: float,
    holding_bars: int,
    commission: float = 0,
    slippage: float = 0,
    use_spread: bool = True
) -> BacktestResult:
    leader_ret = df[f'{leader}_log_return']
    follower_ret = df[f'{follower}_log_return']
    leader_spread = df[f'{leader}_spread']
    follower_spread = df[f'{follower}_spread']
    follower_close = df[f'{follower}_close']

    signals = []
    for i in range(lag_bars, len(df) - holding_bars):
        leader_move = leader_ret.iloc[i - lag_bars]
        if abs(leader_move) > entry_threshold:
            direction = PositionType.LONG if leader_move > 0 else PositionType.SHORT
            signals.append((i, direction))

    trades = []
    for entry_idx, direction in signals:
        exit_idx = min(entry_idx + holding_bars, len(df) - 1)

        entry_price = follower_close.iloc[entry_idx]
        exit_price = follower_close.iloc[exit_idx]

        if direction == PositionType.LONG:
            gross_pnl = exit_price - entry_price
        else:
            gross_pnl = entry_price - exit_price

        spread_cost = 0
        if use_spread:
            spread_cost = calculate_transaction_cost(
                entry_price, exit_price,
                follower_spread.iloc[entry_idx],
                follower_spread.iloc[exit_idx],
                commission, slippage
            )

        net_pnl = gross_pnl - spread_cost

        trades.append(Trade(
            entry_time=df.index[entry_idx],
            exit_time=df.index[exit_idx],
            entry_price=entry_price,
            exit_price=exit_price,
            position_type=direction,
            size=1.0,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            holding_bars=exit_idx - entry_idx
        ))

    return compute_backtest_metrics(trades, df)


def simulate_mean_reversion_strategy(
    df: pd.DataFrame,
    zscore_col: str,
    entry_threshold: float,
    exit_threshold: float,
    max_holding_bars: int,
    commission: float = 0,
    slippage: float = 0,
    use_spread: bool = True
) -> BacktestResult:
    zscore = df[zscore_col]
    silver_close = df['silver_close']
    silver_spread = df['silver_spread']
    gold_close = df['gold_close']
    gold_spread = df['gold_spread']

    trades = []
    position = PositionType.FLAT
    entry_idx = None
    entry_zscore = None

    for i in range(len(df)):
        z = zscore.iloc[i]

        if position == PositionType.FLAT:
            if z >= entry_threshold:
                position = PositionType.SHORT
                entry_idx = i
                entry_zscore = z
            elif z <= -entry_threshold:
                position = PositionType.LONG
                entry_idx = i
                entry_zscore = z

        elif position != PositionType.FLAT:
            should_exit = False
            if position == PositionType.LONG and z >= -exit_threshold:
                should_exit = True
            elif position == PositionType.SHORT and z <= exit_threshold:
                should_exit = True

            if i - entry_idx >= max_holding_bars:
                should_exit = True

            if should_exit:
                entry_price = silver_close.iloc[entry_idx]
                exit_price = silver_close.iloc[i]

                if position == PositionType.LONG:
                    gross_pnl = exit_price - entry_price
                else:
                    gross_pnl = entry_price - exit_price

                spread_cost = 0
                if use_spread:
                    spread_cost = calculate_transaction_cost(
                        entry_price, exit_price,
                        silver_spread.iloc[entry_idx],
                        silver_spread.iloc[i],
                        commission, slippage
                    )

                net_pnl = gross_pnl - spread_cost

                trades.append(Trade(
                    entry_time=df.index[entry_idx],
                    exit_time=df.index[i],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    position_type=position,
                    size=1.0,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    holding_bars=i - entry_idx
                ))

                position = PositionType.FLAT
                entry_idx = None
                entry_zscore = None

    return compute_backtest_metrics(trades, df)


def compute_backtest_metrics(trades: List[Trade], df: pd.DataFrame) -> BacktestResult:
    if not trades:
        return BacktestResult(
            trades=[], total_trades=0, win_rate=0, avg_return=0, median_return=0,
            avg_winner=0, avg_loser=0, profit_factor=0, expectancy=0,
            max_drawdown=0, max_consecutive_losses=0, sharpe_ratio=0, sortino_ratio=0,
            avg_holding_bars=0, best_trade=0, worst_trade=0,
            gross_return=0, total_costs=0, net_return=0, trades_per_day=0, cost_to_edge_ratio=0
        )

    returns = [t.net_pnl for t in trades]
    gross_returns = [t.gross_pnl for t in trades]
    costs = [t.gross_pnl - t.net_pnl for t in trades]

    winners = [r for r in returns if r > 0]
    losers = [r for r in returns if r <= 0]

    total_days = (df.index[-1] - df.index[0]).total_seconds() / 86400

    equity_curve = np.cumsum(returns)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve - running_max
    max_dd = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    consecutive_losses = 0
    max_consecutive = 0
    for r in returns:
        if r <= 0:
            consecutive_losses += 1
            max_consecutive = max(max_consecutive, consecutive_losses)
        else:
            consecutive_losses = 0

    return BacktestResult(
        trades=trades,
        total_trades=len(trades),
        win_rate=len(winners) / len(trades) * 100,
        avg_return=np.mean(returns),
        median_return=np.median(returns),
        avg_winner=np.mean(winners) if winners else 0,
        avg_loser=np.mean(losers) if losers else 0,
        profit_factor=abs(np.sum(winners) / np.sum(losers)) if losers and np.sum(losers) != 0 else np.inf,
        expectancy=np.mean(returns),
        max_drawdown=max_dd,
        max_consecutive_losses=max_consecutive,
        sharpe_ratio=np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 6) if np.std(returns) > 0 else 0,
        sortino_ratio=np.mean(returns) / np.std([r for r in returns if r < 0]) * np.sqrt(252 * 24 * 6) if any(r < 0 for r in returns) else 0,
        avg_holding_bars=np.mean([t.holding_bars for t in trades]),
        best_trade=np.max(returns),
        worst_trade=np.min(returns),
        gross_return=np.sum(gross_returns),
        total_costs=np.sum(costs),
        net_return=np.sum(returns),
        trades_per_day=len(trades) / total_days if total_days > 0 else 0,
        cost_to_edge_ratio=np.sum(costs) / np.sum(gross_returns) if np.sum(gross_returns) != 0 else 0
    )