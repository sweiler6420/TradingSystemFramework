"""
Conditional Value at Risk Measure
=================================

Calculate Conditional Value at Risk (CVaR) at specified confidence level.
"""

import pandas as pd
import numpy as np
from framework.performance.measures import BaseMeasure


class CVaRMeasure(BaseMeasure):
    """
    Calculate Conditional Value at Risk (CVaR) at specified confidence level.
    """
    
    def __init__(self, confidence_level: float = 0.05):
        super().__init__("Conditional Value at Risk")
        self.confidence_level = confidence_level
    
    def calculate(self, returns: pd.Series, **kwargs) -> float:
        confidence_level = kwargs.get('confidence_level', self.confidence_level)
        var = np.percentile(returns, confidence_level * 100)
        return returns[returns <= var].mean()
