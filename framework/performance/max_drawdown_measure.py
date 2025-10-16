"""
Maximum Drawdown Measure
========================

Calculate maximum drawdown from returns.
"""

import pandas as pd
import numpy as np
from framework.performance.measures import BaseMeasure


class MaxDrawdownMeasure(BaseMeasure):
    """Calculate maximum drawdown"""
    
    def __init__(self):
        super().__init__("Maximum Drawdown")
    
    def calculate(self, returns: pd.Series, **kwargs) -> float:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
