from dataclasses import dataclass
from typing import Optional, List
import yaml
from pathlib import Path


@dataclass
class MT5Config:
    gold_symbols: List[str]
    silver_symbols: List[str]
    timeframe: str


@dataclass
class DataConfig:
    bars: int
    include_ticks: bool
    save_csv: bool
    save_parquet: bool


@dataclass
class AnalysisConfig:
    correlation_windows: List[int]
    beta_window: int
    zscore_window: int
    max_lead_lag_bars: int
    divergence_zscore: List[float]


@dataclass
class BacktestConfig:
    enabled: bool
    transaction_cost_model: bool
    slippage_model: bool


@dataclass
class OutputConfig:
    directory: str


@dataclass
class Config:
    mt5: MT5Config
    data: DataConfig
    analysis: AnalysisConfig
    backtest: BacktestConfig
    output: OutputConfig


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)

    return Config(
        mt5=MT5Config(**raw['mt5']),
        data=DataConfig(**raw['data']),
        analysis=AnalysisConfig(**raw['analysis']),
        backtest=BacktestConfig(**raw['backtest']),
        output=OutputConfig(**raw['output'])
    )