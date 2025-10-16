"""
Value at Risk Measure
=====================

Calculate Value at Risk (VaR) at specified confidence level.
"""

import pandas as pd
import numpy as np
from framework.performance.measures import BaseMeasure


class VaRMeasure(BaseMeasure):
    """
    Calculate Value at Risk (VaR) at specified confidence level.
    """
    
    def __init__(self, confidence_level: float = 0.05):
        super().__init__("Value at Risk")
        self.confidence_level = confidence_level
    
    def calculate(self, returns: pd.Series, **kwargs) -> float:
        confidence_level = kwargs.get('confidence_level', self.confidence_level)
        return np.percentile(returns, confidence_level * 100)
