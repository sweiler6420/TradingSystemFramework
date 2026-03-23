"""
Maximum Drawdown Measure
========================

Calculate maximum drawdown from returns.
"""

import polars as pl
import numpy as np
from framework.performance.measures import BaseMeasure


class MaxDrawdownMeasure(BaseMeasure):
    """Calculate maximum drawdown"""
    
    def __init__(self):
        super().__init__("Maximum Drawdown")
    
    def calculate(self, returns: pl.Series, **kwargs) -> float:
        cumulative = (1 + returns).cum_prod()
        running_max = cumulative.cum_max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
