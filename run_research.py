#!/usr/bin/env python3
"""
Gold-Silver Quantitative Research System
Main entry point for running the research analysis.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from src.config import load_config, Config
from src.mt5_client import MT5Client
from src.data_loader import DataLoader
from src.data_cleaner import clean_ohlcv, synchronize_data
from src.features import (
    add_features, calculate_volatility_regime, calculate_log_returns,
    calculate_ema, calculate_rsi, calculate_atr, calculate_adx
)
from src.correlation import (
    calculate_rolling_correlations, correlation_statistics,
    correlation_by_session, correlation_by_regime
)
from src.lead_lag import analyze_lead_lag
from src.ratio_analysis import (
    calculate_ratio, calculate_ratio_features, analyze_ratio_extremes, ratio_reversion_test
)
from src.spread_analysis import (
    calculate_spread_stats, spread_by_session, spread_by_volatility_regime, spread_cost_analysis
)
from src.regime_analysis import add_session_features, regime_analysis
from src.divergence import (
    detect_divergences, analyze_divergence_forward_returns,
    conditional_divergence_analysis
)
from src.statistics import correlation_stability
from src.walkforward import create_walkforward_windows, run_walkforward_analysis, summarize_walkforward
from src.backtest import (
    simulate_lead_lag_strategy, simulate_mean_reversion_strategy
)
from src.reporting import generate_research_report
from src.plotting import generate_all_charts


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('research.log')
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Gold-Silver Quantitative Research System')
    parser.add_argument('--bars', type=int, help='Number of bars to retrieve')
    parser.add_argument('--timeframe', type=str, help='Timeframe (M1, M5, M10, M15, H1, etc.)')
    parser.add_argument('--symbol-gold', type=str, help='Gold symbol override')
    parser.add_argument('--symbol-silver', type=str, help='Silver symbol override')
    parser.add_argument('--ticks', action='store_true', help='Include tick data')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    return parser.parse_args()


def apply_overrides(config: Config, args) -> Config:
    if args.bars:
        config.data.bars = args.bars
    if args.timeframe:
        config.mt5.timeframe = args.timeframe
    if args.symbol_gold:
        config.mt5.gold_symbols = [args.symbol_gold] + config.mt5.gold_symbols
    if args.symbol_silver:
        config.mt5.silver_symbols = [args.symbol_silver] + config.mt5.silver_symbols
    if args.ticks:
        config.data.include_ticks = True
    return config


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("GOLD-SILVER QUANTITATIVE RESEARCH SYSTEM")
    logger.info("=" * 60)

    config = load_config(args.config)
    config = apply_overrides(config, args)

    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    (output_dir / "charts").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)

    with MT5Client(config.mt5) as client:
        if not client.initialize():
            logger.error("Failed to initialize MT5")
            return 1

        gold_symbol, silver_symbol = client.discover_symbols()
        if not gold_symbol or not silver_symbol:
            logger.error("Symbol discovery failed. Check config.yaml for your broker's symbols.")
            return 1

        loader = DataLoader(client, config)
        gold_df, silver_df = loader.load_gold_silver()

        if gold_df is None or silver_df is None:
            logger.error("Failed to load market data")
            return 1

        logger.info("Cleaning and validating data")
        gold_clean, gold_quality = clean_ohlcv(gold_df, "Gold")
        silver_clean, silver_quality = clean_ohlcv(silver_df, "Silver")

        logger.info("Synchronizing datasets")
        sync_df, sync_report = synchronize_data(gold_clean, silver_clean)

        if len(sync_df) < 100:
            logger.error(f"Insufficient synchronized data: {len(sync_df)} bars")
            return 1

        logger.info(f"Synchronized dataset: {len(sync_df)} bars")

        logger.info("Calculating features")
        sync_df = add_features(sync_df, 'gold_')
        sync_df = add_features(sync_df, 'silver_')

        sync_df['gold_log_return'] = calculate_log_returns(sync_df['gold_close'])
        sync_df['silver_log_return'] = calculate_log_returns(sync_df['silver_log_return'])
        sync_df['gold_pct_return'] = sync_df['gold_close'].pct_change()
        sync_df['silver_pct_return'] = sync_df['silver_close'].pct_change()

        sync_df['volatility_regime'] = calculate_volatility_regime(sync_df['gold_log_return'])
        sync_df = add_session_features(sync_df)

        logger.info("Calculating rolling correlations")
        corr_series = calculate_rolling_correlations(sync_df, config.analysis.correlation_windows)
        corr_stats = {w: correlation_statistics(corr_series[w]) for w in config.analysis.correlation_windows}

        logger.info("Calculating Gold/Silver ratio and features")
        sync_df['ratio'] = calculate_ratio(sync_df['gold_close'], sync_df['silver_close'])
        ratio_features = calculate_ratio_features(
            sync_df['ratio'],
            config.analysis.correlation_windows,
            config.analysis.zscore_window
        )
        for col in ratio_features.columns:
            sync_df[col] = ratio_features[col]

        beta_window = config.analysis.beta_window
        sync_df['cov'] = sync_df['gold_log_return'].rolling(beta_window).cov(sync_df['silver_log_return'])
        sync_df['var_gold'] = sync_df['gold_log_return'].rolling(beta_window).var()
        sync_df['beta'] = sync_df['cov'] / sync_df['var_gold']

        sync_df['expected_silver_return'] = sync_df['beta'] * sync_df['gold_log_return']
        sync_df['residual'] = sync_df['silver_log_return'] - sync_df['expected_silver_return']
        sync_df['residual_zscore'] = (
            sync_df['residual'] / sync_df['residual'].rolling(config.analysis.zscore_window).std()
        )

        logger.info("Running lead/lag analysis")
        lead_lag_results = analyze_lead_lag(
            sync_df,
            config.analysis.max_lead_lag_bars,
            'session',
            'volatility_regime'
        )

        logger.info("Running divergence analysis")
        zscore_col = 'residual_zscore'
        forward_periods = [1, 2, 3, 6, 12]
        thresholds = config.analysis.divergence_zscore

        events = detect_divergences(sync_df, zscore_col, thresholds, cooldown_bars=12)
        divergence_results = analyze_divergence_forward_returns(sync_df, events, forward_periods)
        divergence_by_direction = analyze_divergence_by_direction(sync_df, events, forward_periods)

        conditional_div = conditional_divergence_analysis(
            sync_df, zscore_col, 'session',
            ['ASIA', 'LONDON', 'LONDON_NEW_YORK_OVERLAP', 'NEW_YORK'],
            thresholds, forward_periods
        )

        logger.info("Running ratio analysis")
        ratio_extremes = analyze_ratio_extremes(
            sync_df,
            f'ratio_zscore_{config.analysis.zscore_window}',
            thresholds,
            forward_periods
        )
        ratio_reversion = ratio_reversion_test(
            sync_df,
            f'ratio_zscore_{config.analysis.zscore_window}',
            thresholds,
            [0, 0.5, 1.0]
        )

        logger.info("Running regime analysis")
        regime_results = regime_analysis(sync_df)

        logger.info("Running spread analysis")
        gold_spread_stats = calculate_spread_stats(sync_df['gold_spread'])
        silver_spread_stats = calculate_spread_stats(sync_df['silver_spread'])
        spread_session = spread_by_session(sync_df, 'session')
        spread_regime = spread_by_volatility_regime(sync_df, 'volatility_regime')
        spread_costs = spread_cost_analysis(sync_df)

        logger.info("Running statistical tests")
        stationarity_results = {
            'gold_close': adf_test(sync_df['gold_close']),
            'silver_close': adf_test(sync_df['silver_close']),
            'ratio': adf_test(sync_df['ratio']),
            'residual': adf_test(sync_df['residual']),
        }
        coint_result = engle_granger_test(sync_df['gold_close'], sync_df['silver_close'])
        half_life = calculate_half_life(sync_df['ratio'])
        corr_stability = correlation_stability(
            sync_df['gold_log_return'],
            sync_df['silver_log_return'],
            100
        )

        logger.info("Running strategy backtests")
        backtest_results = {}

        if 'best_gold_silver' in lead_lag_results:
            bgs = lead_lag_results['best_gold_silver']
            if bgs.correlation > 0.1:
                bt = simulate_lead_lag_strategy(
                    sync_df, 'gold', 'silver',
                    bgs.lag_bars, 0.0005, 6,
                    commission=0.5, slippage=0.2
                )
                backtest_results['Lead_Lag_Gold_Silver'] = bt

        if 'best_silver_gold' in lead_lag_results:
            bsg = lead_lag_results['best_silver_gold']
            if bsg.correlation > 0.1:
                bt = simulate_lead_lag_strategy(
                    sync_df, 'silver', 'gold',
                    bsg.lag_bars, 0.0005, 6,
                    commission=0.5, slippage=0.2
                )
                backtest_results['Lead_Lag_Silver_Gold'] = bt

        for threshold in [2.0, 2.5]:
            bt = simulate_mean_reversion_strategy(
                sync_df, 'residual_zscore',
                threshold, 0.5, 24,
                commission=0.5, slippage=0.2
            )
            backtest_results[f'Mean_Reversion_Z{threshold}'] = bt

        logger.info("Running walk-forward analysis")
        walkforward_results = {}
        if backtest_results:
            for name, result in backtest_results.items():
                if hasattr(result, 'trades') and result.trades:
                    if 'Lead_Lag' in name:
                        strategy_fn = simulate_lead_lag_strategy
                        strat_params = {'commission': 0.5, 'slippage': 0.2}
                    else:
                        strategy_fn = simulate_mean_reversion_strategy
                        strat_params = {'commission': 0.5, 'slippage': 0.2}
                    
                    windows = create_walkforward_windows(sync_df)
                    if windows:
                        wf_results = run_walkforward_analysis(sync_df, strategy_fn, strat_params, windows)
                        wf_summary = summarize_walkforward(wf_results)
                        walkforward_results[name] = wf_summary

        logger.info("Generating charts")
        generate_all_charts(
            sync_df,
            {'series': corr_series, 'stats': corr_stats},
            lead_lag_results,
            {k: {'upper': v.get('upper', {}), 'lower': v.get('lower', {})} 
             for k, v in divergence_results.items()},
            regime_results,
            output_dir / "charts"
        )

        logger.info("Saving datasets")
        if config.data.save_csv:
            sync_df.to_csv(output_dir / "data" / "gold_silver_m10.csv")
        if config.data.save_parquet:
            sync_df.to_parquet(output_dir / "data" / "gold_silver_m10.parquet")

        quality_reports = {'gold': gold_quality, 'silver': silver_quality}

        correlation_results = {'series': corr_series, 'stats': corr_stats}
        ratio_results = {
            'ratio_stats': type('obj', (object,), {
                'current': sync_df['ratio'].iloc[-1],
                'mean': sync_df['ratio'].mean(),
                'std': sync_df['ratio'].std(),
                'percentile': (sync_df['ratio'] <= sync_df['ratio'].iloc[-1]).mean() * 100,
                'zscore': sync_df[f'ratio_zscore_{config.analysis.zscore_window}'].iloc[-1]
            })(),
            'beta': {
                'mean': sync_df['beta'].mean(),
                'median': sync_df['beta'].median(),
                'min': sync_df['beta'].min(),
                'max': sync_df['beta'].max(),
                'std': sync_df['beta'].std()
            },
            'residual_zscore': {
                'mean': sync_df['residual_zscore'].mean(),
                'std': sync_df['residual_zscore'].std(),
                'min': sync_df['residual_zscore'].min(),
                'max': sync_df['residual_zscore'].max()
            }
        }
        divergence_results_combined = {
            'events': divergence_results,
            'by_direction': divergence_by_direction,
            'conditional': conditional_div
        }
        spread_results = {
            'gold': gold_spread_stats,
            'silver': silver_spread_stats,
            'by_session': spread_session,
            'by_regime': spread_regime,
            'cost_analysis': spread_costs
        }
        stats_results = {
            'stationarity': stationarity_results,
            'cointegration': coint_result,
            'half_life': half_life,
            'correlation_stability': corr_stability
        }

        logger.info("Generating research report")
        generate_research_report(
            sync_df, config,
            correlation_results, lead_lag_results,
            ratio_results, divergence_results,
            regime_results, spread_results,
            stats_results, backtest_results,
            quality_reports, sync_report,
            output_dir / "reports",
            walkforward_results
        )

        logger.info("Research completed successfully!")
        logger.info(f"Output directory: {output_dir.absolute()}")

    return 0


if __name__ == '__main__':
    sys.exit(main())