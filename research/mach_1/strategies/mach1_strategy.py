"""
Mach1Strategy - RSI Mean Reversion Strategy
==========================================

Strategy implementation for Mach 1 research project.
"""

import pandas as pd
import numpy as np
from framework import SignalBasedStrategy, RSIFeature


class Mach1Strategy(SignalBasedStrategy):
    """Simple RSI Mean Reversion Strategy for Mach 1 research"""
    
    def __init__(self, rsi_period=14, oversold=30, overbought=70, long_only=False, **kwargs):
        super().__init__("Mach 1 RSI Strategy", long_only=long_only)
        self.rsi_feature = RSIFeature(period=rsi_period)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_raw_signals(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate RSI mean reversion signals"""
        
        # Override parameters if provided
        rsi_period = kwargs.get('rsi_period', self.rsi_period)
        oversold = kwargs.get('oversold', self.oversold)
        overbought = kwargs.get('overbought', self.overbought)
        
        # Update feature parameters if changed
        if rsi_period != self.rsi_period:
            self.rsi_feature.set_params(period=rsi_period)
            self.rsi_period = rsi_period
        
        # Get RSI values
        rsi_values = self.rsi_feature.calculate(data)
        
        # Create signals
        signals = pd.Series(0, index=data.index)
        signals[rsi_values < oversold] = 1   # Buy when oversold
        signals[rsi_values > overbought] = -1  # Sell when overbought
        
        return signals
