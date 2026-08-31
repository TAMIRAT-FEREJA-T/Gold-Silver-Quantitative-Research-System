import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Dict, Optional


def save_chart(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_normalized_prices(df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 7))

    gold_norm = df['gold_close'] / df['gold_close'].iloc[0] * 100
    silver_norm = df['silver_close'] / df['silver_close'].iloc[0] * 100

    ax.plot(df.index, gold_norm, label='Gold (Normalized)', color='gold', alpha=0.8)
    ax.plot(df.index, silver_norm, label='Silver (Normalized)', color='silver', alpha=0.8)

    ax.set_title('Normalized Gold vs Silver Prices (Base=100)', fontsize=14)
    ax.set_ylabel('Normalized Price', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    save_chart(fig, output_dir / "01_normalized_prices.png")


def plot_ratio(df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    ratio = df['gold_close'] / df['silver_close']
    axes[0].plot(df.index, ratio, color='purple', alpha=0.8)
    axes[0].axhline(ratio.mean(), color='black', linestyle='--', label=f'Mean: {ratio.mean():.2f}')
    axes[0].set_title('Gold/Silver Ratio', fontsize=14)
    axes[0].set_ylabel('Ratio')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if 'ratio_zscore_100' in df.columns:
        zscore = df['ratio_zscore_100']
        axes[1].plot(df.index, zscore, color='red', alpha=0.8)
        axes[1].axhline(2, color='red', linestyle='--', alpha=0.5, label='+2σ')
        axes[1].axhline(-2, color='red', linestyle='--', alpha=0.5, label='-2σ')
        axes[1].axhline(0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_title('Ratio Z-Score (100-bar window)', fontsize=14)
        axes[1].set_ylabel('Z-Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    if 'residual_zscore' in df.columns:
        rz = df['residual_zscore']
        axes[2].plot(df.index, rz, color='blue', alpha=0.8)
        axes[2].axhline(2, color='red', linestyle='--', alpha=0.5, label='+2σ')
        axes[2].axhline(-2, color='red', linestyle='--', alpha=0.5, label='-2σ')
        axes[2].axhline(0, color='black', linestyle='-', alpha=0.3)
        axes[2].set_title('Residual Z-Score (Silver vs Gold Beta)', fontsize=14)
        axes[2].set_ylabel('Z-Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    save_chart(fig, output_dir / "02_ratio_and_zscore.png")


def plot_rolling_correlation(corr_results: Dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ['blue', 'green', 'orange', 'red']
    for i, (window, series) in enumerate(corr_results.items()):
        if isinstance(series, pd.Series):
            ax.plot(series.index, series.values, label=f'{window} bars ({window*10} min)', 
                   color=colors[i % len(colors)], alpha=0.7)

    ax.set_title('Rolling Correlation: Gold vs Silver Returns', fontsize=14)
    ax.set_ylabel('Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    save_chart(fig, output_dir / "03_rolling_correlation.png")


def plot_dynamic_beta(df: pd.DataFrame, output_dir: Path):
    if 'beta' not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df.index, df['beta'], color='green', alpha=0.8)
    ax.axhline(df['beta'].mean(), color='black', linestyle='--', label=f'Mean: {df["beta"].mean():.3f}')
    ax.set_title('Dynamic Beta (Rolling 100-bar)', fontsize=14)
    ax.set_ylabel('Beta (Silver vs Gold)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    save_chart(fig, output_dir / "04_dynamic_beta.png")


def plot_residual_zscore(df: pd.DataFrame, output_dir: Path):
    if 'residual_zscore' not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df.index, df['residual_zscore'], color='blue', alpha=0.8)
    ax.axhline(2, color='red', linestyle='--', alpha=0.5, label='+2σ')
    ax.axhline(-2, color='red', linestyle='--', alpha=0.5, label='-2σ')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_title('Residual Z-Score (Silver Return vs Expected from Gold)', fontsize=14)
    ax.set_ylabel('Z-Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    save_chart(fig, output_dir / "05_residual_zscore.png")


def plot_lead_lag(lead_lag_results: Dict, output_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    gs = lead_lag_results.get('gold_to_silver')
    if gs is not None and not gs.empty:
        axes[0].bar(gs['lag_bars'], gs['correlation'], color='gold', alpha=0.7, edgecolor='black')
        axes[0].set_title('Gold -> Silver Lead/Lag Correlation', fontsize=14)
        axes[0].set_xlabel('Lag (bars)')
        axes[0].set_ylabel('Correlation')
        axes[0].grid(True, alpha=0.3)

    sg = lead_lag_results.get('silver_to_gold')
    if sg is not None and not sg.empty:
        axes[1].bar(sg['lag_bars'], sg['correlation'], color='silver', alpha=0.7, edgecolor='black')
        axes[1].set_title('Silver -> Gold Lead/Lag Correlation', fontsize=14)
        axes[1].set_xlabel('Lag (bars)')
        axes[1].set_ylabel('Correlation')
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_chart(fig, output_dir / "06_lead_lag_correlation.png")


def plot_returns_distribution(df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(df['gold_log_return'].dropna(), bins=100, alpha=0.7, color='gold', edgecolor='black')
    axes[0, 0].set_title('Gold Log Returns Distribution')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(df['silver_log_return'].dropna(), bins=100, alpha=0.7, color='silver', edgecolor='black')
    axes[0, 1].set_title('Silver Log Returns Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(df['gold_log_return'], df['silver_log_return'], alpha=0.3, s=1)
    axes[1, 0].set_title('Gold vs Silver Returns Scatter')
    axes[1, 0].set_xlabel('Gold Return')
    axes[1, 0].set_ylabel('Silver Return')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(df['gold_log_return'], df['silver_log_return'] - df['gold_log_return'], alpha=0.3, s=1)
    axes[1, 1].set_title('Gold Return vs Residual (Silver - Gold)')
    axes[1, 1].set_xlabel('Gold Return')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_chart(fig, output_dir / "07_returns_distribution.png")


def plot_session_analysis(regime_results: Dict, output_dir: Path):
    sessions = {k: v for k, v in regime_results.items() if not k.startswith('vol_')}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    session_names = list(sessions.keys())
    correlations = [sessions[s].get('correlation', 0) for s in session_names]
    axes[0, 0].bar(session_names, correlations, color=['gold', 'orange', 'red', 'blue'])
    axes[0, 0].set_title('Correlation by Session')
    axes[0, 0].set_ylabel('Correlation')
    axes[0, 0].grid(True, alpha=0.3)

    gold_vol = [sessions[s].get('gold_volatility', 0) for s in session_names]
    silver_vol = [sessions[s].get('silver_volatility', 0) for s in session_names]
    x = np.arange(len(session_names))
    width = 0.35
    axes[0, 1].bar(x - width/2, gold_vol, width, label='Gold', color='gold')
    axes[0, 1].bar(x + width/2, silver_vol, width, label='Silver', color='silver')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(session_names)
    axes[0, 1].set_title('Volatility by Session')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    gold_spread = [sessions[s].get('gold_avg_spread', 0) for s in session_names]
    silver_spread = [sessions[s].get('silver_avg_spread', 0) for s in session_names]
    axes[1, 0].bar(x - width/2, gold_spread, width, label='Gold', color='gold')
    axes[1, 0].bar(x + width/2, silver_spread, width, label='Silver', color='silver')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(session_names)
    axes[1, 0].set_title('Average Spread by Session')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    gold_ret = [sessions[s].get('gold_avg_return', 0) * 10000 for s in session_names]
    silver_ret = [sessions[s].get('silver_avg_return', 0) * 10000 for s in session_names]
    axes[1, 1].bar(x - width/2, gold_ret, width, label='Gold', color='gold')
    axes[1, 1].bar(x + width/2, silver_ret, width, label='Silver', color='silver')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(session_names)
    axes[1, 1].set_title('Avg Return (bps) by Session')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_chart(fig, output_dir / "08_session_analysis.png")


def plot_forward_returns(divergence_results: Dict, output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    idx = 0
    for threshold, data in divergence_results.items():
        if idx >= 4:
            break

        periods = []
        upper_means = []
        lower_means = []

        for period_key in sorted(data.get('upper', {}).keys()):
            p = int(period_key.split('_')[0])
            periods.append(p)
            upper_means.append(data['upper'][period_key].get('mean', 0) * 10000)
            lower_means.append(data['lower'][period_key].get('mean', 0) * 10000)

        axes[idx].plot(periods, upper_means, 'r-o', label='Silver Strong (Upper)')
        axes[idx].plot(periods, lower_means, 'b-o', label='Gold Strong (Lower)')
        axes[idx].axhline(0, color='black', linestyle='-', alpha=0.3)
        axes[idx].set_title(f'Forward Returns after Divergence (Z>={threshold})')
        axes[idx].set_xlabel('Forward Period (bars)')
        axes[idx].set_ylabel('Mean Return (bps)')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    for i in range(idx, 4):
        axes[i].axis('off')

    fig.tight_layout()
    save_chart(fig, output_dir / "09_forward_returns_divergence.png")


def generate_all_charts(
    df: pd.DataFrame,
    correlation_results: Dict,
    lead_lag_results: Dict,
    divergence_results: Dict,
    regime_results: Dict,
    output_dir: Path
):
    plot_normalized_prices(df, output_dir)
    plot_ratio(df, output_dir)
    plot_rolling_correlation(correlation_results.get('series', {}), output_dir)
    plot_dynamic_beta(df, output_dir)
    plot_residual_zscore(df, output_dir)
    plot_lead_lag(lead_lag_results, output_dir)
    plot_returns_distribution(df, output_dir)
    plot_session_analysis(regime_results, output_dir)
    plot_forward_returns(divergence_results, output_dir)