"""
Strategy Backtest
=================

Main class for running strategy backtests with all components.
"""

from typing import Dict, Any, Optional


class StrategyBacktest:
    """
    Main class for running strategy backtests with all components.
    """
    
    def __init__(self, strategy, data_handler, optimizer: Optional = None):
        self.strategy = strategy
        self.data_handler = data_handler
        self.optimizer = optimizer
        
        # Set up strategy
        self.strategy.set_data_handler(data_handler)
        if optimizer:
            self.strategy.set_optimizer(optimizer)
    
    def run(self, optimize_first: bool = False, **kwargs) -> Dict[str, Any]:
        """Run the complete backtest"""
        
        # Optimize if requested
        if optimize_first and self.optimizer:
            optimization_results = self.strategy.optimize(**kwargs)
            kwargs.update(optimization_results)
        
        # Run strategy
        results = self.strategy.run_strategy(**kwargs)
        
        # Add significance test
        results['significance_test'] = self.strategy.run_significance_test(**kwargs)
        
        return results
    
    def plot_results(self, results: Dict[str, Any], **kwargs):
        """Plot strategy results"""
        # This would be implemented based on your plotting preferences
        pass
