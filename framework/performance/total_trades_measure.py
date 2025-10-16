"""
Total Trades Measure
====================

Calculate total number of trades.
"""

import pandas as pd
import numpy as np
from framework.performance.measures import BaseMeasure


class TotalTradesMeasure(BaseMeasure):
    """Calculate total number of trades"""
    
    def __init__(self):
        super().__init__("Total Trades")
    
    def calculate(self, returns: pd.Series, **kwargs) -> int:
        return len(returns[returns != 0])
