"""
Win Rate Measure
================

Calculate win rate (percentage of winning trades).
"""

import pandas as pd
import numpy as np
from framework.performance.measures import BaseMeasure


class WinRateMeasure(BaseMeasure):
    """Calculate win rate"""
    
    def __init__(self):
        super().__init__("Win Rate")
    
    def calculate(self, returns: pd.Series, **kwargs) -> float:
        total_trades = len(returns[returns != 0])
        winning_trades = len(returns[returns > 0])
        return winning_trades / total_trades if total_trades > 0 else 0
