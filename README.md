# Trading Strategy Framework with Monte Carlo Permutation Testing

A comprehensive Object-Oriented Programming (OOP) framework for developing, testing, and validating trading strategies with built-in Monte Carlo permutation testing capabilities.

## Overview

This framework provides a modular, extensible architecture for trading strategy development with three core components:
1. **Strategy Implementation** - Indicators, models, features, and signal generation
2. **Data Handling** - Market data loading, preprocessing, and validation
3. **Performance Analysis** - Comprehensive performance measures and Monte Carlo testing

The framework is designed with clean separation of concerns, making it easy to add new strategies, performance measures, and testing methodologies.

## Quick Start

### Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd TradingSystemBacktests
```

2. Create and activate a virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Examples

Run the comprehensive example demonstrating the framework:
```bash
python -m framework.examples.example_usage
```

This will execute:
- RSI strategy with optimization and Monte Carlo testing
- Donchian breakout strategy with statistical validation
- Strategy comparison with performance metrics

## Framework Architecture

### Directory Structure

```
framework/
├── __init__.py                 # Main framework exports
├── strategies/                 # Strategy implementations
│   ├── __init__.py
│   ├── base_strategy.py       # BaseStrategy and Optimizer classes
│   ├── rsi_strategy.py        # RSI-based strategy
│   └── donchian_strategy.py   # Donchian breakout strategy
├── performance/               # Performance analysis tools
│   ├── __init__.py
│   ├── measures.py           # BaseMeasure abstract class
│   ├── returns_measure.py    # Returns calculation
│   ├── profit_factor_measure.py
│   ├── sharpe_ratio_measure.py
│   ├── max_drawdown_measure.py
│   ├── total_return_measure.py
│   ├── win_rate_measure.py
│   ├── total_trades_measure.py
│   ├── monte_carlo_measures.py # Monte Carlo testing
│   ├── calmar_ratio_measure.py # Advanced risk measures
│   ├── sortino_ratio_measure.py
│   ├── var_measure.py
│   └── cvar_measure.py
├── data_handling/            # Data management
│   ├── __init__.py
│   └── data_handler.py      # DataHandler class
├── data_sources/            # Market data providers (yfinance) + Parquet cache
│   ├── __init__.py
│   ├── protocol.py
│   ├── yfinance_provider.py
│   ├── cache.py
│   └── retry.py
├── backtest/                # Backtesting functionality
│   ├── __init__.py
│   └── strategy_backtest.py # StrategyBacktest class
├── examples/                # Usage examples
│   └── example_usage.py    # Comprehensive framework demo
└── data/                   # Sample data files
    ├── BTCUSD1hour.pq
    ├── BTCUSD1min.pq
    └── btcusd_1-min_data.csv
```

## Core Components

### 1. Strategy Module (`framework/strategies/`)

#### BaseStrategy Class
Abstract base class for all trading strategies. Provides:
- Signal generation interface
- Performance calculation integration
- Monte Carlo testing capabilities
- Optimizer integration

```python
from framework.strategies import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("My Custom Strategy")
    
    def generate_signals(self, **kwargs) -> pd.Series:
        # Your signal generation logic
        pass
```

#### Optimizer Class
Abstract base class for strategy optimization:
- Parameter optimization
- Model training
- Pattern selection

```python
from framework.strategies import Optimizer

class MyOptimizer(Optimizer):
    def optimize(self, data, strategy, **kwargs):
        # Your optimization logic
        pass
```

### 2. Performance Module (`framework/performance/`)

#### BaseMeasure Class
Abstract base class for all performance measures:

```python
from framework.performance import BaseMeasure

class CustomMeasure(BaseMeasure):
    def __init__(self):
        super().__init__("Custom Measure")
    
    def calculate(self, data, **kwargs):
        # Your calculation logic
        pass
```

#### Available Performance Measures

**Basic Measures:**
- `ReturnsMeasure` - Calculate strategy returns from signals
- `ProfitFactorMeasure` - Winning trades / losing trades ratio
- `SharpeRatioMeasure` - Risk-adjusted returns
- `MaxDrawdownMeasure` - Maximum portfolio decline
- `TotalReturnMeasure` - Total log return
- `WinRateMeasure` - Percentage of winning trades
- `TotalTradesMeasure` - Total number of trades

**Advanced Risk Measures:**
- `CalmarRatioMeasure` - Annual return / maximum drawdown
- `SortinoRatioMeasure` - Excess return / downside deviation
- `VaRMeasure` - Value at Risk at specified confidence level
- `CVaRMeasure` - Conditional Value at Risk

**Monte Carlo Testing:**
- `MonteCarloPermutationTest` - Statistical significance testing
- Multiple permutation methods (bar permutation, simple, block)

### 3. Data Handling Module (`framework/data_handling/`)

#### DataHandler Class
Handles market data loading, preprocessing, and validation:

```python
from framework.data_handling import DataHandler

# Create data handler
data_handler = DataHandler('path/to/data.pq')

# Load and process data
data_handler.load_data()
data_handler.filter_date_range(2020, 2021)

# Add custom features
data_handler.add_features('rsi', rsi_values)
```

**Features:**
- Supports Parquet (.pq) and CSV files
- Automatic timestamp handling
- Column standardization
- Date range filtering
- Feature addition capabilities
- Crypto market optimized (24/7 trading)

#### Market data providers (`framework/data_sources/`)

Fetch OHLCV via a pluggable **provider**, cache Parquet under each research project’s `data/` folder, then pass the path to `DataHandler`. The default implementation uses **yfinance** with chunked requests, retries, and delays between chunks.

```python
from datetime import date
from framework.data_sources import YFinanceProvider, ensure_cached
from framework.data_handling import DataHandler

cache_dir = "/path/to/research/my_strategy/data"
path = ensure_cached(
    YFinanceProvider(),
    symbol="AAPL",
    interval="1h",
    start=date(2023, 1, 1),
    end=date(2024, 1, 1),
    cache_dir=cache_dir,
)
data_handler = DataHandler(str(path))
data_handler.load_data()
```

### 4. Backtest Module (`framework/backtest/`)

#### StrategyBacktest Class
Main class for running complete strategy backtests:

```python
from framework.backtest import StrategyBacktest

# Create backtest
backtest = StrategyBacktest(strategy, data_handler, optimizer)

# Run with optimization
results = backtest.run(optimize_first=True)

# Access results
performance = results['performance']
monte_carlo = results['monte_carlo']
```

## Usage Examples

### Basic Strategy Implementation

```python
from framework import BaseStrategy, DataHandler, StrategyBacktest
from framework.performance import SharpeRatioMeasure, MaxDrawdownMeasure

# 1. Create strategy
class MyStrategy(BaseStrategy):
    def generate_signals(self, **kwargs):
        # Your signal logic here
        return signals

# 2. Set up data
data_handler = DataHandler('data/BTCUSD1hour.pq')
data_handler.load_data()

# 3. Create and run backtest
strategy = MyStrategy()
backtest = StrategyBacktest(strategy, data_handler)
results = backtest.run()

# 4. Analyze performance
print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']}")
print(f"Max Drawdown: {results['performance']['max_drawdown']}")
```

### Custom Performance Measures

```python
from framework.performance import BaseMeasure
import pandas as pd
import numpy as np

class VolatilityMeasure(BaseMeasure):
    def __init__(self, periods_per_year=252):
        super().__init__("Annualized Volatility")
        self.periods_per_year = periods_per_year
    
    def calculate(self, returns, **kwargs):
        return returns.std() * np.sqrt(self.periods_per_year)

# Use custom measure
vol_measure = VolatilityMeasure(252)
volatility = vol_measure.calculate(returns)
```

### Strategy-Specific Performance Measures

```python
# Conservative strategy measures
from framework.performance import SharpeRatioMeasure, MaxDrawdownMeasure, VaRMeasure

conservative_measures = [
    SharpeRatioMeasure(risk_free_rate=0.02),
    MaxDrawdownMeasure(),
    VaRMeasure(confidence_level=0.01)
]

# Aggressive strategy measures
from framework.performance import CalmarRatioMeasure, SortinoRatioMeasure

aggressive_measures = [
    CalmarRatioMeasure(periods_per_year=252),
    SortinoRatioMeasure(target_return=0.05)
]
```

### Monte Carlo Testing

```python
from framework.performance import MonteCarloPermutationTest

# Create Monte Carlo test
mc_test = MonteCarloPermutationTest(
    n_permutations=1000,
    significance_level=0.05
)

# Run test
results = mc_test.calculate(data, strategy_returns)

print(f"P-value: {results['p_value']}")
print(f"Significant: {results['is_significant']}")
```

## Framework Benefits

* **Modular Design** - Easy to add new strategies and measures
* **Consistent Performance Measurement** - All measures follow the same interface
* **Built-in Monte Carlo Testing** - Statistical significance validation
* **Optimizer Framework** - Parameter selection capabilities
* **Crypto-focused Data Handling** - 24/7 market support
* **OOP Structure** - Maintainable and extensible
* **Clean Separation of Concerns** - Each module has a single responsibility
* **Extensible Architecture** - Easy to add new components

## Advanced Features

### Custom Optimizers
```python
class GridSearchOptimizer(Optimizer):
    def optimize(self, data, strategy, **kwargs):
        # Grid search implementation
        pass
```

### Multiple Data Sources
```python
# Handle multiple markets
data_handlers = [
    DataHandler('data/BTCUSD1hour.pq'),
    DataHandler('data/ETHUSD1hour.pq')
]
```

### Walk-Forward Analysis
```python
# Implement walk-forward testing
for period in date_ranges:
    train_data = data_handler.filter_date_range(period[0], period[1])
    test_data = data_handler.filter_date_range(period[1], period[2])
    # Run strategy on test data
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MIT License
