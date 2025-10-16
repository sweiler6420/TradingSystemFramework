# Trading System Framework - Feature-Based Architecture

## Overview

The framework now uses a feature-based architecture that separates indicators/features from trading strategies. This creates a more modular, extensible system that can be used for both traditional trading strategies and machine learning models.

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    Trading System Framework                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────────────────────────┐  │
│  │   Data Layer    │    │           Features Layer            │  │
│  │                 │    │                                     │  │
│  │ ┌─────────────┐ │    │ ┌─────────────────────────────────┐ │  │
│  │ │ DataHandler │ │──> │ │        BaseFeature              │ │  │
│  │ │             │ │    │ │  (Abstract Base Class)          │ │  │
│  │ │ • Load data │ │    │ │                                 │ │  │
│  │ │ • Filter    │ │    │ │ • calculate()                   │ │  │
│  │ │ • Validate  │ │    │ │ • get_values()                  │ │  │
│  │ └─────────────┘ │    │ │ • set_params()                  │ │  │
│  └─────────────────┘    │ │ • validate_data()               │ │  │
│                         │ └─────────────────────────────────┘ │  │
│                         │                 │                   │  │
│                         │                 ▼                   │  │
│                         │ ┌─────────────────────────────────┐ │  │
│                         │ │         RSIFeature              │ │  │
│                         │ │                                 │ │  │
│                         │ │ • calculate() - RSI values      │ │  │
│                         │ │ • get_momentum_signals()        │ │  │
│                         │ │ • get_divergence_signals()      │ │  │
│                         │ │ • get_overbought_signals()      │ │  │
│                         │ └─────────────────────────────────┘ │  │
│                         │                                     │  │
│                         │ ┌─────────────────────────────────┐ │  │
│                         │ │       DonchianFeature           │ │  │
│                         │ │                                 │ │  │
│                         │ │ • calculate() - middle band     │ │  │
│                         │ │ • get_bands() - all bands       │ │  │
│                         │ │ • get_breakout_signals()        │ │  │
│                         │ │ • get_channel_position()        │ │  │
│                         │ └─────────────────────────────────┘ │  │
│                         └─────────────────────────────────────┘  │
│                                   │                              │
│                                   ▼                              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Strategies Layer                             │ │
│  │                                                             │ │
│  │ ┌─────────────────────────────────────────────────────────┐ │ │
│  │ │                BaseStrategy                             │ │ │
│  │ │                                                         │ │ │
│  │ │ • generate_signals() - Abstract method                  │ │ │
│  │ │ • calculate_performance()                               │ │ │
│  │ │ • run_monte_carlo_test()                                │ │ │
│  │ │ • optimize()                                            │ │ │
│  │ └─────────────────────────────────────────────────────────┘ │ │
│  │                        │                                    │ │
│  │                        ▼                                    │ │
│  │ ┌─────────────────────────────────────────────────────────┐ │ │
│  │ │                RSIStrategy                              │ │ │
│  │ │                                                         │ │ │
│  │ │ • Uses RSIFeature internally                            │ │ │
│  │ │ • generate_signals() - Long-only strategy               │ │ │
│  │ │ • Focuses on trading logic, not RSI calculation         │ │ │
│  │ └─────────────────────────────────────────────────────────┘ │ │
│  │                                                             │ │
│  │ ┌─────────────────────────────────────────────────────────┐ │ │
│  │ │              DonchianStrategy                           │ │ │
│  │ │                                                         │ │ │
│  │ │ • Uses DonchianFeature internally                       │ │ │
│  │ │ • generate_signals() - Long/short strategy              │ │ │
│  │ │ • Focuses on trading logic, not indicator calculation   │ │ │
│  │ └─────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Performance & Backtesting                      │ │
│  │                                                             │ │
│  │ • StrategyBacktest - Orchestrates the process               │ │
│  │ • Performance measures (Sharpe, Drawdown, etc.)             │ │
│  │ • Monte Carlo testing                                       │ │
│  │ • Optimization framework                                    │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Key Benefits

### 1. **Separation of Concerns**
- **Features**: Focus on indicator calculation and signal generation
- **Strategies**: Focus on trading logic and position management
- **Data**: Focus on data loading, filtering, and validation

### 2. **Modularity**
- Features can be used independently
- Easy to add new features without changing existing code
- Strategies can combine multiple features
- Features can be reused across different strategies

### 3. **ML-Ready Architecture**
- Features provide clean, consistent interfaces
- Easy to create feature datasets for machine learning
- Features can be used for both traditional strategies and ML models
- Rich feature information and validation

### 4. **Extensibility**
- Add new features by extending `BaseFeature`
- Add new strategies by extending `BaseStrategy`
- Features can be composed and combined
- Easy to add new signal types and analysis methods

## Usage Examples

### Using Features Directly
```python
from framework.features import RSIFeature, DonchianFeature

# Create features
rsi_feature = RSIFeature(period=14, overbought=70, oversold=30)
donchian_feature = DonchianFeature(lookback=20)

# Calculate values
rsi_values = rsi_feature.calculate(data)
donchian_bands = donchian_feature.get_bands(data)

# Get signals
rsi_signals = rsi_feature.get_momentum_signals(data)
breakout_signals = donchian_feature.get_breakout_signals(data)
```

### Using Features in Strategies
```python
from framework.strategies import RSIStrategy, DonchianStrategy

# Strategies internally use features
rsi_strategy = RSIStrategy(rsi_period=14, buy_threshold=20, sell_threshold=80)
donchian_strategy = DonchianStrategy(lookback=20)

# Generate trading signals
signals = rsi_strategy.generate_signals()
```

### Creating ML Datasets
```python
# Create feature dataset for ML
ml_data = data.copy()
ml_data['rsi'] = rsi_feature.calculate(data)
ml_data['rsi_oversold'] = rsi_feature.get_oversold_signals(data)
ml_data['donchian_breakout'] = donchian_feature.get_breakout_signals(data)['upper_breakout']
```

## Adding New Features

To add a new feature, simply extend `BaseFeature`:

```python
from framework.features.base_feature import BaseFeature

class MovingAverageFeature(BaseFeature):
    def __init__(self, period=20):
        super().__init__("MovingAverage", period=period)
        self.period = period
    
    def calculate(self, data):
        return data['close'].rolling(window=self.period).mean()
    
    def get_signals(self, data):
        # Add custom signal logic
        pass
```

## Future Extensions

This architecture supports many future enhancements:

1. **More Features**: Bollinger Bands, MACD, Stochastic, etc.
2. **Feature Composition**: Combine multiple features
3. **ML Integration**: Direct integration with scikit-learn, PyTorch, etc.
4. **Feature Selection**: Automatic feature selection for ML models
5. **Feature Engineering**: Automated feature creation and optimization
6. **Real-time Features**: Streaming feature calculation
7. **Feature Validation**: Advanced validation and quality checks
