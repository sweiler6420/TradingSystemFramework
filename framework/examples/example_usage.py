"""
Example Usage of Trading Strategy Framework
==========================================

This example shows how to use the OOP framework for trading strategies.
"""

from framework.data_handling import DataHandler
from framework.backtest import StrategyBacktest
from framework.strategies.rsi_strategy import RSIStrategy, RSIOptimizer
from framework.strategies.donchian_strategy import DonchianStrategy, DonchianOptimizer


def run_rsi_example():
    """Example of running RSI strategy with optimization and Monte Carlo testing"""
    
    print("=== RSI Strategy Example ===")
    
    # 1. Set up data handler
    data_handler = DataHandler('framework/data/BTCUSD1hour.pq')
    data_handler.load_data()
    data_handler.filter_date_range(2020, 2021)
    
    # 2. Create strategy
    rsi_strategy = RSIStrategy(rsi_period=14, buy_threshold=20, sell_threshold=80)
    
    # 3. Set up optimizer
    rsi_optimizer = RSIOptimizer()
    
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
    donchian_strategy = DonchianStrategy(lookback=20)
    
    # 3. Set up optimizer
    donchian_optimizer = DonchianOptimizer()
    
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


def compare_strategies():
    """Compare multiple strategies"""
    
    print("\n=== Strategy Comparison ===")
    
    # Set up data
    data_handler = DataHandler('framework/data/BTCUSD1hour.pq')
    data_handler.load_data()
    data_handler.filter_date_range(2020, 2021)
    
    # Create strategies
    strategies = {
        'RSI': RSIStrategy(rsi_period=14, buy_threshold=20, sell_threshold=80),
        'Donchian': DonchianStrategy(lookback=20)
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
    comparison_results = compare_strategies()
    
    print("\n=== Framework Benefits ===")
    print("* Modular design - easy to add new strategies")
    print("* Consistent performance measurement")
    print("* Built-in Monte Carlo testing")
    print("* Optimizer framework for parameter selection")
    print("* Crypto-focused data handling")
    print("* OOP structure for maintainability")
