"""
Base Strategy Classes
====================

Contains the abstract base classes for trading strategies.
"""

from abc import ABC, abstractmethod
import polars as pl
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import warnings


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    def __init__(self, name: str, data: pl.DataFrame):
        """
        Initialize base strategy with data.
        
        Args:
            name: Strategy name
            data: Market data DataFrame (required for all strategies)
        """
        self.name = name
        self.data = data
        self.data_handler = None
        self.signals = None
        self.returns = None
        self.performance = {}
        

    def set_data_handler(self, data_handler):
        """Set the data handler"""
        self.data_handler = data_handler
        self.data = data_handler.get_data()
        

    @abstractmethod
    def generate_signals(self, **kwargs) -> pl.Series:
        """Generate trading signals"""
        pass
    

    def calculate_performance(self) -> Dict[str, Any]:
        """Calculate performance measures"""
        from framework.performance.returns_measure import ReturnsMeasure
        from framework.performance.profit_factor_measure import ProfitFactorMeasure
        from framework.performance.sharpe_ratio_measure import SharpeRatioMeasure
        from framework.performance.total_return_measure import TotalReturnMeasure
        from framework.performance.max_drawdown_measure import MaxDrawdownMeasure
        from framework.performance.win_rate_measure import WinRateMeasure
        from framework.performance.total_trades_measure import TotalTradesMeasure
        
        if self.returns is None:
            returns_measure = ReturnsMeasure('signal')
            self.returns = returns_measure.calculate(self.data)
            
        # Create measure instances
        profit_factor_measure = ProfitFactorMeasure()
        sharpe_measure = SharpeRatioMeasure()
        total_return_measure = TotalReturnMeasure()
        max_drawdown_measure = MaxDrawdownMeasure()
        win_rate_measure = WinRateMeasure()
        total_trades_measure = TotalTradesMeasure()
        
        self.performance = {
            'profit_factor': profit_factor_measure.calculate(self.returns),
            'sharpe_ratio': sharpe_measure.calculate(self.returns),
            'total_return': total_return_measure.calculate(self.returns),
            'max_drawdown': max_drawdown_measure.calculate(self.returns),
            'total_trades': total_trades_measure.calculate(self.returns),
            'win_rate': win_rate_measure.calculate(self.returns)
        }
        
        return self.performance
    

    def run_significance_test(self, significance_test=None, **kwargs) -> Dict[str, Any]:
        """Run significance test to validate strategy results"""
        from framework.significance_testing.monte_carlo_significance_test import MonteCarloSignificanceTest
        
        if significance_test is None:
            significance_test = MonteCarloSignificanceTest()
        
        if self.returns is None:
            self.calculate_performance()
            
        return significance_test.get_results(self.data, self.returns, **kwargs)
    

    def create_custom_plots(self, data: pl.DataFrame, signal_result, **kwargs) -> list:
        """
        Override this method in your strategy to create custom plots.
        Return a list of plot objects that will be embedded in the performance plots.
        
        Args:
            data: The market data DataFrame
            signal_result: The signal result object
            **kwargs: Additional parameters
            
        Returns:
            List of plot objects (Bokeh figures, matplotlib plots, etc.)
        """
        return []  # Default: no custom plots
    
    def run_strategy(self, **kwargs) -> Dict[str, Any]:
        """Run the complete strategy pipeline"""
        # Generate signals
        self.signals = self.generate_signals(**kwargs)
        self.data = self.data.with_columns(self.signals.alias('signal'))
        
        # Calculate performance
        performance = self.calculate_performance()
        
        return {
            'strategy_name': self.name,
            'performance': performance,
            'signals': self.signals,
            'data': self.data
        }
