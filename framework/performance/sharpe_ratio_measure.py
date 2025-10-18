"""
Sharpe Ratio Measure
====================

Calculate Sharpe ratio (excess return / volatility).
"""

import polars as pl
import numpy as np
from framework.performance.measures import BaseMeasure


class SharpeRatioMeasure(BaseMeasure):
    """Calculate Sharpe ratio"""
    
    def __init__(self, risk_free_rate: float = 0.0):
        super().__init__("Sharpe Ratio")
        self.risk_free_rate = risk_free_rate
    
    def calculate(self, returns: pl.Series, **kwargs) -> float:
        risk_free_rate = kwargs.get('risk_free_rate', self.risk_free_rate)
        excess_returns = returns.mean() - risk_free_rate
        return excess_returns / returns.std() if returns.std() > 0 else 0
