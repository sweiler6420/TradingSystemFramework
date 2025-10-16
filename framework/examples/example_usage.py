"""
Example Usage of Trading Strategy Framework
==========================================

This example shows how to use the OOP framework for trading strategies
with the new feature-based architecture.
"""

from framework.data_handling import DataHandler
from framework.backtest import StrategyBacktest
from framework.strategies.rsi_breakout_strategy import RSIBreakoutStrategy, RSIBreakoutOptimizer
from framework.strategies.donchian_breakout_strategy import DonchianBreakoutStrategy, DonchianBreakoutOptimizer
from framework.features.rsi_feature import RSIFeature
from framework.features.donchian_feature import DonchianFeature


def run_rsi_example():
    """Example of running RSI strategy with optimization and Monte Carlo testing"""
    
    print("=== RSI Strategy Example ===")
    
    # 1. Set up data handler
    data_handler = DataHandler('framework/data/BTCUSD1hour.pq')
    data_handler.load_data()
    data_handler.filter_date_range(2020, 2021)
    
    # 2. Create strategy
    rsi_strategy = RSIBreakoutStrategy(rsi_period=14, buy_threshold=20, sell_threshold=80)
    
    # 3. Set up optimizer
    rsi_optimizer = RSIBreakoutOptimizer()
    
    # 4. Create backtest
    backtest = StrategyBacktest(rsi_strategy, data_handler, rsi_optimizer)
    
    # 5. Run optimization and backtest
    results = backtest.run(optimize_first=True, 
                          rsi_periods=[10, 14, 21], 
                          buy_thresholds=[15, 20, 25], 
                          sell_thresholds=[75, 80, 85])
    
    # 6. Display results
    print(f"Strategy: {results['strategy_name']}")
    print(f"Performance: {results['performance']}")
    print(f"Monte Carlo P-value: {results['monte_carlo']['p_value']:.4f}")
    print(f"Strategy is significant: {results['monte_carlo']['is_significant']}")
    
    return results


def run_donchian_example():
    """Example of running Donchian strategy"""
    
    print("\n=== Donchian Strategy Example ===")
    
    # 1. Set up data handler
    data_handler = DataHandler('framework/data/BTCUSD1hour.pq')
    data_handler.load_data()
    data_handler.filter_date_range(2020, 2021)
    
    # 2. Create strategy
    donchian_strategy = DonchianBreakoutStrategy(lookback=20)
    
    # 3. Set up optimizer
    donchian_optimizer = DonchianBreakoutOptimizer()
    
    # 4. Create backtest
    backtest = StrategyBacktest(donchian_strategy, data_handler, donchian_optimizer)
    
    # 5. Run optimization and backtest
    results = backtest.run(optimize_first=True, 
                          lookback_range=range(15, 50))
    
    # 6. Display results
    print(f"Strategy: {results['strategy_name']}")
    print(f"Performance: {results['performance']}")
    print(f"Monte Carlo P-value: {results['monte_carlo']['p_value']:.4f}")
    print(f"Strategy is significant: {results['monte_carlo']['is_significant']}")
    
    return results


def demonstrate_features():
    """Demonstrate using features directly for analysis and ML"""
    
    print("\n=== Feature-Based Analysis Example ===")
    
    # Set up data
    data_handler = DataHandler('framework/data/BTCUSD1hour.pq')
    data_handler.load_data()
    data_handler.filter_date_range(2020, 2021)
    data = data_handler.get_data()
    
    # Create features
    rsi_feature = RSIFeature(period=14, overbought=70, oversold=30)
    donchian_feature = DonchianFeature(lookback=20)
    
    # Calculate feature values
    rsi_values = rsi_feature.calculate(data)
    donchian_bands = donchian_feature.get_bands(data)
    
    # Get various signals from features
    rsi_momentum = rsi_feature.get_momentum_signals(data)
    rsi_divergence = rsi_feature.get_divergence_signals(data)
    donchian_breakouts = donchian_feature.get_breakout_signals(data)
    donchian_position = donchian_feature.get_channel_position(data)
    
    # Display feature information
    print(f"RSI Feature Info: {rsi_feature.get_feature_info()}")
    print(f"Donchian Feature Info: {donchian_feature.get_feature_info()}")
    
    # Show some statistics
    print(f"\nRSI Statistics:")
    print(f"  Mean RSI: {rsi_values.mean():.2f}")
    print(f"  Oversold periods: {rsi_momentum['oversold'].sum()}")
    print(f"  Overbought periods: {rsi_momentum['overbought'].sum()}")
    print(f"  Bullish divergences: {rsi_divergence['bullish_divergence'].sum()}")
    
    print(f"\nDonchian Statistics:")
    print(f"  Mean channel width: {donchian_feature.get_channel_width(data).mean():.2f}")
    print(f"  Upper breakouts: {donchian_breakouts['upper_breakout'].sum()}")
    print(f"  Lower breakdowns: {donchian_breakouts['lower_breakdown'].sum()}")
    print(f"  Mean position in channel: {donchian_position.mean():.3f}")
    
    # Example: Create a simple feature-based dataset for ML
    feature_data = data.copy()
    feature_data['rsi'] = rsi_values
    feature_data['rsi_overbought'] = rsi_momentum['overbought']
    feature_data['rsi_oversold'] = rsi_momentum['oversold']
    feature_data['donchian_upper'] = donchian_bands['upper']
    feature_data['donchian_lower'] = donchian_bands['lower']
    feature_data['donchian_position'] = donchian_position
    feature_data['donchian_breakout'] = donchian_breakouts['upper_breakout']
    
    print(f"\nFeature Dataset Shape: {feature_data.shape}")
    print("Features available for ML:")
    feature_cols = ['rsi', 'rsi_overbought', 'rsi_oversold', 'donchian_upper', 
                   'donchian_lower', 'donchian_position', 'donchian_breakout']
    for col in feature_cols:
        print(f"  - {col}")
    
    return feature_data


def compare_strategies():
    """Compare multiple strategies"""
    
    print("\n=== Strategy Comparison ===")
    
    # Set up data
    data_handler = DataHandler('framework/data/BTCUSD1hour.pq')
    data_handler.load_data()
    data_handler.filter_date_range(2020, 2021)
    
    # Create strategies
    strategies = {
        'RSI': RSIBreakoutStrategy(rsi_period=14, buy_threshold=20, sell_threshold=80),
        'Donchian': DonchianBreakoutStrategy(lookback=20)
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        backtest = StrategyBacktest(strategy, data_handler)
        results[name] = backtest.run()
        
        print(f"\n{name} Strategy:")
        print(f"  Profit Factor: {results[name]['performance']['profit_factor']:.4f}")
        print(f"  Sharpe Ratio: {results[name]['performance']['sharpe_ratio']:.4f}")
        print(f"  Total Return: {results[name]['performance']['total_return']:.4f}")
        print(f"  Max Drawdown: {results[name]['performance']['max_drawdown']:.4f}")
        print(f"  Monte Carlo P-value: {results[name]['monte_carlo']['p_value']:.4f}")
    
    return results


if __name__ == "__main__":
    # Run examples
    rsi_results = run_rsi_example()
    donchian_results = run_donchian_example()
    feature_data = demonstrate_features()
    comparison_results = compare_strategies()
    
    print("\n=== Framework Benefits ===")
    print("* Modular design - easy to add new strategies")
    print("* Feature-based architecture - indicators separated from strategies")
    print("* Consistent performance measurement")
    print("* Built-in Monte Carlo testing")
    print("* Optimizer framework for parameter selection")
    print("* Crypto-focused data handling")
    print("* OOP structure for maintainability")
    print("* ML-ready features for model training")
