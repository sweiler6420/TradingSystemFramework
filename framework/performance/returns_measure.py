"""
Returns Measure
===============

Calculate strategy returns from signals.
"""

import polars as pl
import numpy as np
from framework.performance.measures import BaseMeasure


class ReturnsMeasure(BaseMeasure):
    """Calculate strategy returns from signals"""
    
    def __init__(self, signal_col: str = 'signal'):
        super().__init__("Returns Calculation")
        self.signal_col = signal_col
    
    def calculate(self, data: pl.DataFrame, **kwargs) -> pl.Series:
        signal_col = kwargs.get('signal_col', self.signal_col)
        if 'return' not in data.columns:
            data = data.with_columns(
                pl.col('close').log().diff().shift(-1).alias('return')
            )
        return data.select(pl.col(signal_col) * pl.col('return'))[signal_col]
