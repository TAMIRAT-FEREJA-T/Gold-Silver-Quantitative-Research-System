# Gold–Silver Quantitative Research System

A **research-only** quantitative analysis application for studying the relationship between Gold (XAUUSD) and Silver (XAGUSD) using historical data from MetaTrader 5.

## ⚠️ Important Disclaimer

**This is NOT a trading bot.** The system does NOT:
- Place, modify, or close orders
- Manage positions
- Execute live trades
- Connect to trading APIs beyond read-only market data

This is strictly a **statistical research and analysis tool**. All outputs are for research purposes only.

## What This Project Does

- Retrieves historical M10 (10-minute) OHLCV data from your local MT5 terminal
- Synchronizes Gold and Silver data by timestamp
- Calculates returns, correlations, ratios, and advanced features
- Performs lead/lag analysis (Gold → Silver and Silver → Gold)
- Analyzes Gold/Silver ratio and residual Z-scores
- Tests mean-reversion and momentum strategy candidates
- Generates comprehensive statistical reports and visualizations
- Implements proper out-of-sample testing and look-ahead bias prevention

## What This Project Does NOT Do

- Execute trades
- Manage risk in real-time
- Connect to broker trading APIs
- Guarantee profitable strategies
- Replace proper backtesting infrastructure for live trading

## Installation

### Prerequisites

- Windows with MetaTrader 5 installed
- Python 3.11+
- MT5 terminal must be running and logged in

### Setup

```bash
# Clone or navigate to the project
cd gold-silver_research

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### MT5 Symbol Configuration

The system auto-discovers your broker's Gold and Silver symbols. Common variations:
- `XAUUSD`, `XAUUSDm`, `XAUUSDc`, `XAUUSD.`, `GOLD`, `GOLDm`
- `XAGUSD`, `XAGUSDm`, `XAGUSDc`, `XAGUSD.`, `SILVER`, `SILVERm`

If your broker uses different symbols, add them to `config.yaml`:

```yaml
mt5:
  gold_symbols:
    - XAUUSD
    - YOUR_BROKER_GOLD_SYMBOL
  silver_symbols:
    - XAGUSD
    - YOUR_BROKER_SILVER_SYMBOL
```

## Usage

### Basic Run

```bash
python run_research.py
```

### With Overrides

```bash
# Custom number of bars
python run_research.py --bars 10000

# Custom timeframe
python run_research.py --timeframe M15

# Override symbols
python run_research.py --symbol-gold XAUUSDm --symbol-silver XAGUSDm

# Verbose logging
python run_research.py --verbose
```

### First Run Recommendation

Start with a small dataset for testing:

```bash
python run_research.py --bars 10000
```

After validation, run full research:

```bash
python run_research.py --bars 50000
```

## Output Files

All outputs are saved to `output/`:

```
output/
├── data/
│   ├── gold_silver_m10.csv      # Full synchronized dataset with features
│   └── gold_silver_m10.parquet  # Parquet format (faster loading)
├── charts/
│   ├── 01_normalized_prices.png
│   ├── 02_ratio_and_zscore.png
│   ├── 03_rolling_correlation.png
│   ├── 04_dynamic_beta.png
│   ├── 05_residual_zscore.png
│   ├── 06_lead_lag_correlation.png
│   ├── 07_returns_distribution.png
│   ├── 08_session_analysis.png
│   └── 09_forward_returns_divergence.png
├── reports/
│   ├── research_report.txt      # Full text report
│   └── run_metadata.json        # Run configuration for reproducibility
└── tables/
    └── (CSV exports for further analysis)
```

## Key Analyses

### 1. Rolling Correlation
- Windows: 20, 50, 100, 200 bars (200min, 500min, 1000min, 2000min)
- Statistics: mean, median, std, % positive, % > 0.5, % > 0.7

### 2. Gold/Silver Ratio
- Current ratio, rolling mean/std, percentile, Z-score
- Extreme threshold analysis (±1.5, ±2.0, ±2.5, ±3.0)

### 3. Dynamic Beta (Hedge Ratio)
- Rolling 100-bar beta: cov(Gold, Silver) / var(Gold)
- Beta stability across sessions and volatility regimes

### 4. Residual Z-Score
- Expected Silver return = Beta × Gold return
- Residual = Actual Silver return - Expected
- Z-score identifies relative-value divergences

### 5. Lead/Lag Analysis
- Tests 0-120 minutes (0-12 bars) in both directions
- Gold → Silver and Silver → Gold
- Conditional analysis by session and volatility regime

### 6. Divergence Research
- Detects extreme residual Z-scores
- Measures forward returns after divergence
- Tests reversion probability and timing

### 7. Strategy Candidates
- **Lead/Lag Momentum**: Trade follower after leader moves
- **Mean Reversion**: Trade extreme residual Z-score reversion
- Transaction-cost-aware (spread, commission, slippage)

### 8. Statistical Tests
- Augmented Dickey-Fuller (stationarity)
- Engle-Granger (cointegration)
- Half-life of mean reversion
- Correlation stability

## Interpreting Results

### Correlation ≠ Profitability
High correlation between Gold and Silver returns does **not** imply a tradable relationship. Consider:
- Transaction costs (spread + commission + slippage)
- Execution latency
- Correlation stability over time
- Lead/lag consistency

### Lead/Lag ≠ Predictive Power
A lagged correlation between Gold and Silver may be:
- Spurious (data mining)
- Non-stationary (changes over time)
- Too small to overcome costs
- Not actionable in real-time

### Z-Score Extremes ≠ Guaranteed Reversion
Mean reversion in ratio/residual Z-scores:
- May not exist (unit root)
- May have long half-life
- May revert only in specific regimes
- Can diverge further before reverting

## Methodology Safeguards

### Look-Ahead Bias Prevention
- All signals use only data available at timestamp `t`
- No future returns, volatility, or correlation used
- Rolling calculations use `.shift(1)` where appropriate

### Out-of-Sample Testing
- Chronological splits: 60% research / 20% validation / 20% test
- No shuffling of time-series data
- Final test set never used for parameter tuning

### Walk-Forward Analysis
- Rolling train/validate/test windows
- Reports performance stability across windows

### Overlapping Signal Handling
- Cooldown periods between signals
- Non-overlapping event analysis option
- Both reported for comparison

## Configuration

All parameters configurable via `config.yaml` without code changes:

```yaml
mt5:
  gold_symbols: [...]
  silver_symbols: [...]
  timeframe: M10

data:
  bars: 50000
  include_ticks: false
  save_csv: true
  save_parquet: true

analysis:
  correlation_windows: [20, 50, 100, 200]
  beta_window: 100
  zscore_window: 100
  max_lead_lag_bars: 12
  divergence_zscore: [1.5, 2.0, 2.5, 3.0]

backtest:
  enabled: true
  transaction_cost_model: true
  slippage_model: true

output:
  directory: output
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_features.py -v

# Run with coverage
pytest --cov=src tests/
```

## Project Structure

```
gold-silver_research/
├── run_research.py          # Main entry point
├── config.yaml              # Configuration
├── requirements.txt         # Dependencies
├── README.md               # This file
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration loading
│   ├── mt5_client.py       # MT5 connection & symbol discovery
│   ├── data_loader.py      # Data retrieval
│   ├── data_cleaner.py     # Data validation & synchronization
│   ├── features.py         # Technical indicators & returns
│   ├── correlation.py      # Rolling & conditional correlation
│   ├── lead_lag.py         # Lead/lag analysis
│   ├── ratio_analysis.py   # Ratio & residual analysis
│   ├── spread_analysis.py  # Spread & transaction costs
│   ├── regime_analysis.py  # Session & volatility regimes
│   ├── divergence.py       # Divergence detection & analysis
│   ├── statistics.py       # Statistical tests
│   ├── backtest.py         # Research backtests
│   ├── reporting.py        # Report generation
│   └── plotting.py         # Visualization
├── tests/
│   ├── test_data_loader.py
│   ├── test_features.py
│   ├── test_lead_lag.py
│   └── test_statistics.py
└── output/                 # Generated outputs
```

## Reproducibility

Every run saves `output/reports/run_metadata.json` with:
- Run timestamp
- Symbols used
- Timeframe and bar count
- Analysis parameters
- Software version

## Common Issues

### MT5 Connection Failed
- Ensure MT5 terminal is running
- Check "Allow WebRequest for listed URL" in MT5 Tools → Options → Expert Advisors
- Verify Python can access MT5 (run as administrator if needed)

### Symbols Not Found
- Add your broker's exact symbol names to `config.yaml`
- Check Market Watch in MT5 for correct spelling
- Symbols are case-insensitive but must match exactly

### Insufficient History
- Some brokers limit historical data
- Try smaller `--bars` value
- Check if M10 timeframe is available (some brokers only have M1, M5, M15, H1)

### Memory Issues with Large Datasets
- Use `--bars 10000` for testing
- Parquet format uses less memory than CSV
- Consider using tick data selectively

## Citation

If you use this research system in academic or professional work, please acknowledge:

> Gold-Silver Quantitative Research System, v1.0.0
> https://github.com/your-repo/gold-silver_research

## License

MIT License - See LICENSE file for details.

## Disclaimer

**Past performance does not guarantee future results.** This research system is for educational and analytical purposes only. No representation is made that any strategy identified will be profitable. Trading financial instruments carries substantial risk of loss and is not suitable for all investors.