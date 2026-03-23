"""
Total Return Measure
====================

Calculate total log return.
"""

import polars as pl
import numpy as np
from framework.performance.measures import BaseMeasure


class TotalReturnMeasure(BaseMeasure):
    """Calculate total log return"""
    
    def __init__(self):
        super().__init__("Total Return")
    
    def calculate(self, returns: pl.Series, **kwargs) -> float:
        return returns.sum()
