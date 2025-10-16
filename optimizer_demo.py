"""
Optimizer Architecture Demo
===========================

This script demonstrates the improved optimizer architecture with
feature-level and strategy-level optimizers.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'framework'))

from framework.features.rsi_feature import RSIFeature
from framework.features.donchian_feature import DonchianFeature
from framework.strategies.rsi_breakout_strategy import RSIBreakoutStrategy
from framework.strategies.donchian_breakout_strategy import DonchianBreakoutStrategy
from framework.data_handling import DataHandler
from framework.optimization.feature_optimizer import RSIFeatureOptimizer, DonchianFeatureOptimizer
from framework.optimization.strategy_optimizer import RSIBreakoutStrategyOptimizer, DonchianBreakoutStrategyOptimizer


def demo_feature_optimization():
    """Demonstrate feature-level optimization"""
    
    print("=== Feature-Level Optimization Demo ===\n")
    
    # Load data
    data_handler = DataHandler('framework/data/BTCUSD1hour.pq')
    data_handler.load_data()
    data_handler.filter_date_range(2020, 2021)
    data = data_handler.get_data()
    
    print(f"Loaded data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}\n")
    
    # 1. RSI Feature Optimization
    print("1. RSI Feature Optimization:")
    print("-" * 40)
    
    rsi_feature = RSIFeature(period=14, overbought=70, oversold=30)
    rsi_optimizer = RSIFeatureOptimizer()
    
    # Optimize RSI parameters
    rsi_best_params = rsi_optimizer.optimize_feature_params(
        data, rsi_feature,
        rsi_periods=[10, 14, 21],
        buy_thresholds=[15, 20, 25],
        sell_thresholds=[75, 80, 85],
        optimization_metric='profit_factor'
    )
    
    print(f"Best RSI Parameters:")
    for key, value in rsi_best_params.items():
        print(f"  {key}: {value}")
    
    # 2. Donchian Feature Optimization
    print(f"\n2. Donchian Feature Optimization:")
    print("-" * 40)
    
    donchian_feature = DonchianFeature(lookback=20)
    donchian_optimizer = DonchianFeatureOptimizer()
    
    # Optimize Donchian parameters
    donchian_best_params = donchian_optimizer.optimize_feature_params(
        data, donchian_feature,
        lookback_periods=[15, 20, 25, 30],
        upper_thresholds=[1.0, 1.01],
        lower_thresholds=[0.99, 1.0],
        optimization_metric='profit_factor'
    )
    
    print(f"Best Donchian Parameters:")
    for key, value in donchian_best_params.items():
        print(f"  {key}: {value}")


def demo_strategy_optimization():
    """Demonstrate strategy-level optimization"""
    
    print("\n=== Strategy-Level Optimization Demo ===\n")
    
    # Load data
    data_handler = DataHandler('framework/data/BTCUSD1hour.pq')
    data_handler.load_data()
    data_handler.filter_date_range(2020, 2021)
    data = data_handler.get_data()
    
    # 1. RSI Strategy Optimization
    print("1. RSI Strategy Optimization:")
    print("-" * 40)
    
    rsi_strategy = RSIBreakoutStrategy(rsi_period=14, buy_threshold=20, sell_threshold=80)
    rsi_strategy_optimizer = RSIBreakoutStrategyOptimizer()
    
    # Optimize strategy
    rsi_strategy_params = rsi_strategy_optimizer.optimize_strategy(
        data, rsi_strategy,
        rsi_periods=[10, 14, 21],
        buy_thresholds=[15, 20, 25],
        sell_thresholds=[75, 80, 85],
        optimization_metric='profit_factor'
    )
    
    print(f"Best RSI Strategy Parameters:")
    for key, value in rsi_strategy_params.items():
        print(f"  {key}: {value}")
    
    # 2. Donchian Strategy Optimization
    print(f"\n2. Donchian Strategy Optimization:")
    print("-" * 40)
    
    donchian_strategy = DonchianBreakoutStrategy(lookback=20)
    donchian_strategy_optimizer = DonchianBreakoutStrategyOptimizer()
    
    # Optimize strategy
    donchian_strategy_params = donchian_strategy_optimizer.optimize_strategy(
        data, donchian_strategy,
        lookback_periods=[15, 20, 25, 30],
        upper_thresholds=[1.0, 1.01],
        lower_thresholds=[0.99, 1.0],
        optimization_metric='profit_factor'
    )
    
    print(f"Best Donchian Strategy Parameters:")
    for key, value in donchian_strategy_params.items():
        print(f"  {key}: {value}")


def demo_optimizer_benefits():
    """Demonstrate the benefits of the new optimizer architecture"""
    
    print("\n=== Optimizer Architecture Benefits ===\n")
    
    print("1. Modularity:")
    print("   ✓ Feature optimizers can be reused across strategies")
    print("   ✓ Easy to add new features with their own optimizers")
    print("   ✓ Clear separation between feature and strategy optimization")
    
    print("\n2. Flexibility:")
    print("   ✓ Different optimization metrics per feature")
    print("   ✓ Feature optimizers can be used independently")
    print("   ✓ Strategy optimizers coordinate feature optimization")
    
    print("\n3. Extensibility:")
    print("   ✓ New features automatically get optimizers")
    print("   ✓ Easy to add multi-feature strategies")
    print("   ✓ Support for complex optimization scenarios")
    
    print("\n4. Testability:")
    print("   ✓ Feature optimizers can be tested independently")
    print("   ✓ Clear interfaces make testing easier")
    print("   ✓ Strategy optimizers can use mock feature optimizers")
    
    print("\n5. Performance:")
    print("   ✓ Feature-specific optimization algorithms")
    print("   ✓ Parallel optimization of independent features")
    print("   ✓ Caching of feature calculations")


def demo_usage_comparison():
    """Compare old vs new optimizer usage"""
    
    print("\n=== Usage Comparison ===\n")
    
    print("OLD APPROACH (Strategy-specific optimizers):")
    print("-" * 50)
    print("""
# Old way - optimizer tightly coupled to strategy
class RSIOptimizer(Optimizer):
    def optimize(self, data, strategy, **kwargs):
        # Strategy-specific optimization logic
        # Hard to reuse, tightly coupled
        pass

# Usage
rsi_strategy = RSIStrategy()
rsi_optimizer = RSIOptimizer()  # Strategy-specific
best_params = rsi_optimizer.optimize(data, rsi_strategy)
""")
    
    print("NEW APPROACH (Feature + Strategy optimizers):")
    print("-" * 50)
    print("""
# New way - modular, reusable optimizers
class RSIFeatureOptimizer(FeatureOptimizer):
    def optimize_feature_params(self, data, feature, **kwargs):
        # Feature-specific optimization logic
        # Reusable across any strategy using RSI
        pass

class RSIBreakoutStrategyOptimizer(StrategyOptimizer):
    def __init__(self):
        self.rsi_optimizer = RSIFeatureOptimizer()  # Reusable!
    
    def optimize_strategy(self, data, strategy, **kwargs):
        # Coordinates feature optimization
        # Can add strategy-specific optimization
        pass

# Usage
rsi_strategy = RSIBreakoutStrategy()
strategy_optimizer = RSIBreakoutStrategyOptimizer()
best_params = strategy_optimizer.optimize_strategy(data, rsi_strategy)
""")


if __name__ == "__main__":
    demo_feature_optimization()
    demo_strategy_optimization()
    demo_optimizer_benefits()
    demo_usage_comparison()
    
    print("\n=== Summary ===")
    print("The new optimizer architecture provides:")
    print("• Better separation of concerns")
    print("• Improved reusability and modularity")
    print("• Easier testing and maintenance")
    print("• Clear path for future enhancements")
    print("• Support for complex multi-feature strategies")
