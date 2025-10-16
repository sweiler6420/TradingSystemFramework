"""
Total Return Measure
====================

Calculate total log return.
"""

import pandas as pd
import numpy as np
from framework.performance.measures import BaseMeasure


class TotalReturnMeasure(BaseMeasure):
    """Calculate total log return"""
    
    def __init__(self):
        super().__init__("Total Return")
    
    def calculate(self, returns: pd.Series, **kwargs) -> float:
        return returns.sum()
