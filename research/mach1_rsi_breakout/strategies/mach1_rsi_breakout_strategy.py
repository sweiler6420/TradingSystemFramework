"""
Mach1RsiBreakoutStrategy - RSI Breakout Strategy
===============================================

RSI breakout strategy that enters long positions when RSI breaks above oversold levels
and exits when RSI breaks below overbought levels.
"""

import polars as pl
import numpy as np
from framework import SignalBasedStrategy, RSIFeature, SignalChange


class Mach1RsiBreakoutStrategy(SignalBasedStrategy):
    """RSI Breakout Strategy - Enter on oversold breakout, exit on overbought breakout"""
    
    def __init__(self, data: pl.DataFrame, rsi_period=14, oversold=30, overbought=70, **kwargs):
        super().__init__("Mach1 RSI Breakout Strategy", data)
        # Create RSI feature with data at initialization
        self.rsi_feature = RSIFeature(data, period=rsi_period, oversold=oversold, overbought=overbought)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.position = 0  # Track current position
    
    def generate_raw_signal(self, **kwargs) -> pl.Series:
        """Generate RSI breakout signals using SignalChange enums"""
        
        # Override parameters if provided
        rsi_period = kwargs.get('rsi_period', self.rsi_period)
        oversold = kwargs.get('oversold', self.oversold)
        overbought = kwargs.get('overbought', self.overbought)
        
        # Get RSI values from our stateful feature
        rsi_values = self.rsi_feature.get_values()
        
        # Create signals for breakout strategy using SignalChange enums
        signals_list = [SignalChange.NO_CHANGE] * len(self.data)
        
        # RSI Breakout Logic:
        # Enter long when RSI breaks above oversold level (coming out of oversold)
        # Exit long when RSI breaks below overbought level (coming out of overbought)
        
        for i in range(1, len(self.data)):
            current_rsi = rsi_values[i]
            previous_rsi = rsi_values[i-1]
            
            # Skip if RSI values are NaN
            if current_rsi is None or previous_rsi is None or str(current_rsi) == 'nan' or str(previous_rsi) == 'nan':
                continue
            
            # Enter long position: RSI was oversold and now breaks above oversold
            if (previous_rsi <= oversold and current_rsi > oversold and self.position == 0):
                signals_list[i] = SignalChange.NEUTRAL_TO_LONG
                self.position = 1
            
            # Exit long position: RSI was overbought and now breaks below overbought
            elif (previous_rsi >= overbought and current_rsi < overbought and self.position == 1):
                signals_list[i] = SignalChange.LONG_TO_NEUTRAL
                self.position = 0
        
        # Convert list to Polars Series
        signals = pl.Series(signals_list)
        return signals
    
    def create_custom_plots(self, data: pl.DataFrame, signal_result, **kwargs) -> list:
        """Create custom plots using feature's built-in plot methods"""
        plots = []
        
        # Use our stateful RSI feature's plot method
        rsi_plot = self.rsi_feature.get_plot()
        if rsi_plot is not None:
            plots.append(rsi_plot)
        
        return plots