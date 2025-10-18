"""
Profit Factor Measure
=====================

Calculate profit factor (winning trades / losing trades).
"""

import polars as pl
import numpy as np
from framework.performance.measures import BaseMeasure


class ProfitFactorMeasure(BaseMeasure):
    """Calculate profit factor"""
    
    def __init__(self):
        super().__init__("Profit Factor")
    
    def calculate(self, returns: pl.Series, **kwargs) -> float:
        winning_trades = returns.filter(returns > 0).sum()
        losing_trades = returns.filter(returns < 0).abs().sum()
        return winning_trades / losing_trades if losing_trades > 0 else float('inf')
