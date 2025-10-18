"""
Mach1RsiBreakoutStrategy - RSI Breakout Strategy
===============================================

RSI breakout strategy that enters long positions when RSI breaks above oversold levels
and exits when RSI breaks below overbought levels.
"""

import pandas as pd
import numpy as np
from framework import SignalBasedStrategy, RSIFeature, SignalChange


class Mach1RsiBreakoutStrategy(SignalBasedStrategy):
    """RSI Breakout Strategy - Enter on oversold breakout, exit on overbought breakout"""
    
    def __init__(self, rsi_period=14, oversold=30, overbought=70, long_only=True, **kwargs):
        super().__init__("Mach1 RSI Breakout Strategy", long_only=long_only)
        self.rsi_feature = RSIFeature(period=rsi_period)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.position = 0  # Track current position
    
    def generate_raw_signals(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate RSI breakout signals using SignalChange enums"""
        
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
        
        # Create signals for breakout strategy using SignalChange enums
        signals = pd.Series(SignalChange.NO_CHANGE, index=data.index)
        
        # RSI Breakout Logic:
        # Enter long when RSI breaks above oversold level (coming out of oversold)
        # Exit long when RSI breaks below overbought level (coming out of overbought)
        
        for i in range(1, len(data)):
            current_rsi = rsi_values.iloc[i]
            previous_rsi = rsi_values.iloc[i-1]
            
            # Enter long position: RSI was oversold and now breaks above oversold
            if (previous_rsi <= oversold and current_rsi > oversold and self.position == 0):
                signals.iloc[i] = SignalChange.NEUTRAL_TO_LONG
                self.position = 1
            
            # Exit long position: RSI was overbought and now breaks below overbought
            elif (previous_rsi >= overbought and current_rsi < overbought and self.position == 1):
                signals.iloc[i] = SignalChange.LONG_TO_NEUTRAL
                self.position = 0
        
        return signals