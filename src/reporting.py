import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from src.config import Config
from src.walkforward import create_walkforward_windows, run_walkforward_analysis, summarize_walkforward


def generate_research_report(
    df: pd.DataFrame,
    config: Config,
    correlation_results: Dict,
    lead_lag_results: Dict,
    ratio_results: Dict,
    divergence_results: Dict,
    regime_results: Dict,
    spread_results: Dict,
    stats_results: Dict,
    backtest_results: Dict,
    quality_reports: Dict,
    sync_report: Dict,
    output_dir: Path,
    walkforward_results: Dict = None
):
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("GOLD-SILVER QUANTITATIVE RESEARCH REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    report_lines.append("## DATASET")
    report_lines.append(f"Gold Symbol: {config.mt5.gold_symbols[0]} (discovered: {config.mt5.gold_symbols[0]})")
    report_lines.append(f"Silver Symbol: {config.mt5.silver_symbols[0]} (discovered: {config.mt5.silver_symbols[0]})")
    report_lines.append(f"Timeframe: {config.mt5.timeframe}")
    report_lines.append(f"Bars Requested: {config.data.bars}")
    report_lines.append(f"Gold Bars Retrieved: {quality_reports.get('gold', {}).total_bars if 'gold' in quality_reports else 'N/A'}")
    report_lines.append(f"Silver Bars Retrieved: {quality_reports.get('silver', {}).total_bars if 'silver' in quality_reports else 'N/A'}")
    report_lines.append(f"Matched Bars: {sync_report.get('matched_bars', 'N/A')}")
    report_lines.append(f"Unmatched Gold: {sync_report.get('unmatched_gold', 'N/A')}")
    report_lines.append(f"Unmatched Silver: {sync_report.get('unmatched_silver', 'N/A')}")
    report_lines.append(f"Start Date: {df.index[0] if len(df) > 0 else 'N/A'}")
    report_lines.append(f"End Date: {df.index[-1] if len(df) > 0 else 'N/A'}")
    report_lines.append(f"Missing Periods: {len(sync_report.get('missing_periods', []))}")
    report_lines.append("")

    report_lines.append("## RELATIONSHIP ANALYSIS")
    report_lines.append("### Rolling Correlation")
    for window, stats in correlation_results.get('stats', {}).items():
        report_lines.append(f"  {window} bars ({window * 10} min): mean={stats.mean:.4f}, median={stats.median:.4f}, "
                           f"std={stats.std:.4f}, >0: {stats.pct_positive:.1f}%, >0.5: {stats.pct_above_05:.1f}%, >0.7: {stats.pct_above_07:.1f}%")
    report_lines.append("")

    report_lines.append("### Gold/Silver Ratio")
    if 'ratio_stats' in ratio_results:
        rs = ratio_results['ratio_stats']
        report_lines.append(f"  Current: {rs.current:.4f}")
        report_lines.append(f"  Mean: {rs.mean:.4f}")
        report_lines.append(f"  Std: {rs.std:.4f}")
        report_lines.append(f"  Current Z-Score: {rs.zscore:.4f}")
    report_lines.append("")

    report_lines.append("### Dynamic Beta (Rolling)")
    if 'beta' in ratio_results:
        beta_stats = ratio_results['beta']
        report_lines.append(f"  Mean Beta: {beta_stats.get('mean', 0):.4f}")
        report_lines.append(f"  Median Beta: {beta_stats.get('median', 0):.4f}")
        report_lines.append(f"  Min Beta: {beta_stats.get('min', 0):.4f}")
        report_lines.append(f"  Max Beta: {beta_stats.get('max', 0):.4f}")
        report_lines.append(f"  Beta Std: {beta_stats.get('std', 0):.4f}")
    report_lines.append("")

    report_lines.append("### Residual Z-Score")
    if 'residual_zscore' in ratio_results:
        rz = ratio_results['residual_zscore']
        report_lines.append(f"  Mean: {rz.get('mean', 0):.4f}")
        report_lines.append(f"  Std: {rz.get('std', 0):.4f}")
        report_lines.append(f"  Min: {rz.get('min', 0):.4f}")
        report_lines.append(f"  Max: {rz.get('max', 0):.4f}")
    report_lines.append("")

    report_lines.append("## LEAD/LAG ANALYSIS")
    if 'best_gold_silver' in lead_lag_results:
        bgs = lead_lag_results['best_gold_silver']
        report_lines.append(f"### Gold -> Silver")
        report_lines.append(f"  Best Lag: {bgs.lag_bars} bars ({bgs.lag_minutes} minutes)")
        report_lines.append(f"  Correlation: {bgs.correlation:.4f}")
        report_lines.append(f"  Sample Size: {bgs.sample_size}")
    report_lines.append("")

    if 'best_silver_gold' in lead_lag_results:
        bsg = lead_lag_results['best_silver_gold']
        report_lines.append(f"### Silver -> Gold")
        report_lines.append(f"  Best Lag: {bsg.lag_bars} bars ({bsg.lag_minutes} minutes)")
        report_lines.append(f"  Correlation: {bsg.correlation:.4f}")
        report_lines.append(f"  Sample Size: {bsg.sample_size}")
    report_lines.append("")

    report_lines.append("## SESSION ANALYSIS")
    for session, data in regime_results.items():
        if session.startswith('vol_'):
            continue
        report_lines.append(f"### {session}")
        report_lines.append(f"  Observations: {data.get('observations', 0)}")
        report_lines.append(f"  Correlation: {data.get('correlation', 0):.4f}")
        report_lines.append(f"  Gold Avg Return: {data.get('gold_avg_return', 0):.6f}")
        report_lines.append(f"  Silver Avg Return: {data.get('silver_avg_return', 0):.6f}")
        report_lines.append(f"  Gold Volatility: {data.get('gold_volatility', 0):.6f}")
        report_lines.append(f"  Silver Volatility: {data.get('silver_volatility', 0):.6f}")
        report_lines.append(f"  Gold Avg Spread: {data.get('gold_avg_spread', 0):.2f}")
        report_lines.append(f"  Silver Avg Spread: {data.get('silver_avg_spread', 0):.2f}")
    report_lines.append("")

    report_lines.append("## VOLATILITY REGIME ANALYSIS")
    for regime, data in regime_results.items():
        if not regime.startswith('vol_'):
            continue
        report_lines.append(f"### {regime}")
        report_lines.append(f"  Observations: {data.get('observations', 0)}")
        report_lines.append(f"  Correlation: {data.get('correlation', 0):.4f}")
        report_lines.append(f"  Gold Avg Return: {data.get('gold_avg_return', 0):.6f}")
        report_lines.append(f"  Silver Avg Return: {data.get('silver_avg_return', 0):.6f}")
        report_lines.append(f"  Gold Volatility: {data.get('gold_volatility', 0):.6f}")
        report_lines.append(f"  Silver Volatility: {data.get('silver_volatility', 0):.6f}")
    report_lines.append("")

    report_lines.append("## DIVERGENCE ANALYSIS")
    for threshold, data in divergence_results.items():
        report_lines.append(f"### Z-Score Threshold: {threshold}")
        for direction in ['upper', 'lower']:
            if direction in data:
                for period, stats in data[direction].items():
                    report_lines.append(f"  {direction.capitalize()} {period}: "
                                       f"mean={stats.get('mean', 0):.6f}, "
                                       f"median={stats.get('median', 0):.6f}, "
                                       f"pct_pos={stats.get('pct_positive', 0):.1f}%, "
                                       f"count={stats.get('count', 0)}")
    report_lines.append("")

    report_lines.append("## STRATEGY CANDIDATES")
    for name, result in backtest_results.items():
        report_lines.append(f"### {name}")
        report_lines.append(f"  Total Trades: {result.total_trades}")
        report_lines.append(f"  Win Rate: {result.win_rate:.1f}%")
        report_lines.append(f"  Avg Return: {result.avg_return:.6f}")
        report_lines.append(f"  Median Return: {result.median_return:.6f}")
        report_lines.append(f"  Profit Factor: {result.profit_factor:.2f}")
        report_lines.append(f"  Expectancy: {result.expectancy:.6f}")
        report_lines.append(f"  Max Drawdown: {result.max_drawdown:.6f}")
        report_lines.append(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        report_lines.append(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
        report_lines.append(f"  Avg Holding Bars: {result.avg_holding_bars:.1f}")
        report_lines.append(f"  Best Trade: {result.best_trade:.6f}")
        report_lines.append(f"  Worst Trade: {result.worst_trade:.6f}")
        report_lines.append(f"  Gross Return: {result.gross_return:.6f}")
        report_lines.append(f"  Total Costs: {result.total_costs:.6f}")
        report_lines.append(f"  Net Return: {result.net_return:.6f}")
        report_lines.append(f"  Trades/Day: {result.trades_per_day:.2f}")
        report_lines.append(f"  Cost-to-Edge Ratio: {result.cost_to_edge_ratio:.2f}")
    report_lines.append("")

    report_lines.append("## WALK-FORWARD ANALYSIS")
    if backtest_results:
        for name, result in backtest_results.items():
            if hasattr(result, 'trades') and result.trades:
                from src.walkforward import create_walkforward_windows, run_walkforward_analysis, summarize_walkforward
                from src.backtest import simulate_lead_lag_strategy, simulate_mean_reversion_strategy
                
                if 'Lead_Lag' in name:
                    strategy_fn = simulate_lead_lag_strategy
                else:
                    strategy_fn = simulate_mean_reversion_strategy
                
                windows = create_walkforward_windows(df)
                if windows:
                    wf_results = run_walkforward_analysis(df, strategy_fn, {}, windows)
                    wf_summary = summarize_walkforward(wf_results)
                    report_lines.append(f"### {name} Walk-Forward")
                    report_lines.append(f"  Windows: {wf_summary.get('num_windows', 0)}")
                    report_lines.append(f"  Consistency: {wf_summary.get('consistency', 0):.1f}%")
                    report_lines.append(f"  Avg Net Return: {wf_summary.get('avg_net_return', 0):.6f}")
                    report_lines.append(f"  Avg Sharpe: {wf_summary.get('avg_sharpe', 0):.2f}")
                    report_lines.append(f"  Positive Windows: {wf_summary.get('positive_windows', 0)}")
    report_lines.append("")

    report_lines.append("## STATISTICAL TESTS")
    if 'stationarity' in stats_results:
        for name, result in stats_results['stationarity'].items():
            report_lines.append(f"### {name} Stationarity (ADF)")
            report_lines.append(f"  Statistic: {result.adf_statistic:.4f}")
            report_lines.append(f"  P-Value: {result.p_value:.4f}")
            report_lines.append(f"  Stationary: {result.is_stationary}")
    if 'cointegration' in stats_results:
        c = stats_results['cointegration']
        report_lines.append(f"### Gold-Silver Cointegration")
        report_lines.append(f"  Statistic: {c.coint_statistic:.4f}")
        report_lines.append(f"  P-Value: {c.p_value:.4f}")
        report_lines.append(f"  Cointegrated: {c.is_cointegrated}")
    if 'half_life' in stats_results:
        hl = stats_results['half_life']
        report_lines.append(f"### Spread Half-Life")
        report_lines.append(f"  Half-Life (bars): {hl.half_life:.1f}")
        report_lines.append(f"  Mean Reversion Speed: {hl.mean_reversion_speed:.6f}")
    report_lines.append("")

    report_lines.append("## SPREAD ANALYSIS")
    for name, stats in spread_results.items():
        if isinstance(stats, dict):
            report_lines.append(f"### {name}")
            report_lines.append(f"  Mean: {stats.get('mean', 0):.2f}")
            report_lines.append(f"  Median: {stats.get('median', 0):.2f}")
            report_lines.append(f"  Min: {stats.get('min', 0):.2f}")
            report_lines.append(f"  Max: {stats.get('max', 0):.2f}")
            report_lines.append(f"  95th Pct: {stats.get('pct_95', 0):.2f}")
            report_lines.append(f"  99th Pct: {stats.get('pct_99', 0):.2f}")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("RESEARCH CONCLUSION")
    report_lines.append("=" * 80)

    conclusions = generate_conclusions(
        correlation_results, lead_lag_results, ratio_results,
        divergence_results, regime_results, backtest_results, stats_results
    )

    for i, conclusion in enumerate(conclusions, 1):
        report_lines.append(f"{i}. {conclusion}")

    report_text = "\n".join(report_lines)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "research_report.txt", 'w') as f:
        f.write(report_text)

    metadata = {
        'run_timestamp': datetime.now().isoformat(),
        'gold_symbol': config.mt5.gold_symbols[0],
        'silver_symbol': config.mt5.silver_symbols[0],
        'timeframe': config.mt5.timeframe,
        'bars_requested': config.data.bars,
        'bars_actual': len(df),
        'analysis_params': {
            'correlation_windows': config.analysis.correlation_windows,
            'beta_window': config.analysis.beta_window,
            'zscore_window': config.analysis.zscore_window,
            'max_lead_lag_bars': config.analysis.max_lead_lag_bars,
            'divergence_zscore': config.analysis.divergence_zscore
        }
    }
    with open(output_dir / "run_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return report_text


def generate_conclusions(
    correlation_results: Dict,
    lead_lag_results: Dict,
    ratio_results: Dict,
    divergence_results: Dict,
    regime_results: Dict,
    backtest_results: Dict,
    stats_results: Dict
) -> list[str]:
    conclusions = []

    corr_stats = correlation_results.get('stats', {})
    avg_corr = np.mean([s.mean for s in corr_stats.values()]) if corr_stats else 0
    pct_above_50 = np.mean([s.pct_above_05 for s in corr_stats.values()]) if corr_stats else 0

    if avg_corr > 0.5:
        conclusions.append(f"Gold and Silver are significantly correlated at M10 (avg correlation: {avg_corr:.3f})")
    else:
        conclusions.append(f"Gold and Silver show weak correlation at M10 (avg correlation: {avg_corr:.3f})")

    if pct_above_50 > 50:
        conclusions.append("The relationship is reasonably stable (correlation > 0.5 for >50% of observations)")
    else:
        conclusions.append("The relationship is unstable (correlation > 0.5 for <50% of observations)")

    bgs = lead_lag_results.get('best_gold_silver')
    bsg = lead_lag_results.get('best_silver_gold')

    if bgs and bgs.correlation > 0.1:
        conclusions.append(f"Gold leads Silver at {bgs.lag_bars*10} minutes (correlation: {bgs.correlation:.3f})")
    else:
        conclusions.append("No statistically meaningful Gold->Silver lead/lag relationship found")

    if bsg and bsg.correlation > 0.1:
        conclusions.append(f"Silver leads Gold at {bsg.lag_bars*10} minutes (correlation: {bsg.correlation:.3f})")
    else:
        conclusions.append("No statistically meaningful Silver->Gold lead/lag relationship found")

    if bgs and bsg:
        if bgs.correlation > bsg.correlation:
            strongest = f"Gold->Silver ({bgs.lag_bars*10} min, corr={bgs.correlation:.3f})"
        else:
            strongest = f"Silver->Gold ({bsg.lag_bars*10} min, corr={bsg.correlation:.3f})"
        conclusions.append(f"Strongest lead/lag: {strongest}")
    else:
        conclusions.append("No clear lead/lag relationship identified")

    if 'residual_zscore' in ratio_results:
        rz = ratio_results['residual_zscore']
        if rz.get('std', 0) > 1.5:
            conclusions.append("Residual Z-score shows sufficient variation for mean-reversion analysis")
        else:
            conclusions.append("Residual Z-score shows limited variation")

    if divergence_results:
        total_events = sum(
            data.get('upper', {}).get('count', 0) + data.get('lower', {}).get('count', 0)
            for data in divergence_results.values()
        )
        if total_events > 20:
            conclusions.append(f"Divergence events occur with reasonable frequency ({total_events} events)")
        else:
            conclusions.append(f"Divergence events are rare ({total_events} events)")

    if stats_results.get('cointegration', {}).get('is_cointegrated'):
        conclusions.append("Gold and Silver prices are cointegrated (supports mean-reversion)")
    else:
        conclusions.append("Gold and Silver prices are NOT cointegrated (mean-reversion not statistically supported)")

    if 'half_life' in stats_results:
        hl = stats_results['half_life']
        if hl.half_life < 100:
            conclusions.append(f"Spread half-life is {hl.half_life:.0f} bars (fast mean reversion)")
        else:
            conclusions.append(f"Spread half-life is {hl.half_life:.0f} bars (slow mean reversion)")

    profitable_strategies = [
        name for name, result in backtest_results.items()
        if result.net_return > 0 and result.profit_factor > 1.2
    ]
    if profitable_strategies:
        conclusions.append(f"Strategy candidates with positive net expectancy: {', '.join(profitable_strategies)}")
    else:
        conclusions.append("No strategy candidate shows positive net expectancy after transaction costs")

    if not profitable_strategies:
        conclusions.append("No statistically convincing tradable relationship was identified.")
        conclusions.append("Further research with different timeframes, features, or market regimes is recommended.")

    return conclusions