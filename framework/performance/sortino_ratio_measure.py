"""
Sortino Ratio Measure
=====================

Calculate Sortino ratio (excess return / downside deviation).
"""

import polars as pl
import numpy as np
from framework.performance.measures import BaseMeasure


class SortinoRatioMeasure(BaseMeasure):
    """
    Calculate Sortino ratio (excess return / downside deviation).
    """
    
    def __init__(self, risk_free_rate: float = 0.0, target_return: float = 0.0):
        super().__init__("Sortino Ratio")
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
    
    def calculate(self, returns: pl.Series, **kwargs) -> float:
        risk_free_rate = kwargs.get('risk_free_rate', self.risk_free_rate)
        target_return = kwargs.get('target_return', self.target_return)
        
        excess_returns = returns.mean() - risk_free_rate
        
        # Calculate downside deviation (only negative deviations from target)
        downside_returns = returns.filter(returns < target_return) - target_return
        downside_deviation = np.sqrt((downside_returns ** 2).mean()) if len(downside_returns) > 0 else 0
        
        return excess_returns / downside_deviation if downside_deviation > 0 else 0
