"""
Total Trades Measure
====================

Calculate total number of trades.
"""

import polars as pl
import numpy as np
from framework.performance.measures import BaseMeasure


class TotalTradesMeasure(BaseMeasure):
    """Calculate total number of trades"""
    
    def __init__(self):
        super().__init__("Total Trades")
    
    def calculate(self, returns: pl.Series, **kwargs) -> int:
        return len(returns.filter(returns != 0))
