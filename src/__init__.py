"""
Gold-Silver Quantitative Research System
Research-only analysis of Gold (XAUUSD) and Silver (XAGUSD) relationship.
"""

from src.config import Config, load_config
from src.mt5_client import MT5Client
from src.data_loader import DataLoader
from src.data_cleaner import clean_ohlcv, synchronize_data
from src.features import add_features, calculate_volatility_regime, calculate_log_returns
from src.correlation import calculate_rolling_correlations, correlation_statistics
from src.lead_lag import analyze_lead_lag, find_optimal_lag
from src.ratio_analysis import calculate_ratio, calculate_ratio_features, ratio_reversion_test
from src.spread_analysis import calculate_spread_stats, spread_by_session, spread_cost_analysis
from src.regime_analysis import add_session_features, regime_analysis
from src.divergence import detect_divergences, analyze_divergence_forward_returns
from src.statistics import adf_test, engle_granger_test, calculate_half_life
from src.backtest import simulate_lead_lag_strategy, simulate_mean_reversion_strategy
from src.reporting import generate_research_report
from src.plotting import generate_all_charts

__version__ = "1.0.0"
__all__ = [
    "Config", "load_config",
    "MT5Client",
    "DataLoader",
    "clean_ohlcv", "synchronize_data",
    "add_features", "calculate_volatility_regime", "calculate_log_returns",
    "calculate_rolling_correlations", "correlation_statistics",
    "analyze_lead_lag", "find_optimal_lag",
    "calculate_ratio", "calculate_ratio_features", "ratio_reversion_test",
    "calculate_spread_stats", "spread_by_session", "spread_cost_analysis",
    "add_session_features", "regime_analysis",
    "detect_divergences", "analyze_divergence_forward_returns",
    "adf_test", "engle_granger_test", "calculate_half_life",
    "simulate_lead_lag_strategy", "simulate_mean_reversion_strategy",
    "generate_research_report",
    "generate_all_charts",
]