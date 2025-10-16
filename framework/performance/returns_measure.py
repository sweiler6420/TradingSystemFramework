"""
Returns Measure
===============

Calculate strategy returns from signals.
"""

import pandas as pd
import numpy as np
from framework.performance.measures import BaseMeasure


class ReturnsMeasure(BaseMeasure):
    """Calculate strategy returns from signals"""
    
    def __init__(self, signal_col: str = 'signal'):
        super().__init__("Returns Calculation")
        self.signal_col = signal_col
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        signal_col = kwargs.get('signal_col', self.signal_col)
        if 'return' not in data.columns:
            data['return'] = np.log(data['close']).diff().shift(-1)
        return data[signal_col] * data['return']
