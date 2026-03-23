"""
Calmar Ratio Measure
====================

Calculate Calmar ratio (annual return / maximum drawdown).
"""

import polars as pl
import numpy as np
from framework.performance.measures import BaseMeasure


class CalmarRatioMeasure(BaseMeasure):
    """
    Calculate Calmar ratio (annual return / maximum drawdown).
    """
    
    def __init__(self, periods_per_year: int = 252):
        super().__init__("Calmar Ratio")
        self.periods_per_year = periods_per_year
    
    def calculate(self, returns: pl.Series, **kwargs) -> float:
        periods_per_year = kwargs.get('periods_per_year', self.periods_per_year)
        
        # Calculate annual return
        total_return = returns.sum()
        years = len(returns) / periods_per_year
        annual_return = total_return / years if years > 0 else 0
        
        # Calculate maximum drawdown
        cumulative = (1 + returns).cum_prod()
        running_max = cumulative.cum_max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        return annual_return / max_drawdown if max_drawdown > 0 else 0
