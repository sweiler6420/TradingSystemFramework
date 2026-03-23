"""
Mach2RsiTestingStrategy - Mach2_Rsi_Testing Strategy
=======================================================

Strategy implementation for mach2_rsi_testing research project.
"""

import polars as pl
import numpy as np
from framework import SignalBasedStrategy, PositionState, RSIFeature


class Mach2RsiTestingStrategy(SignalBasedStrategy):
    """Strategy implementation for mach2_rsi_testing research"""
    
    def __init__(self, **kwargs):
        super().__init__("Mach2_Rsi_Testing")
        # Initialize your strategy parameters here
        self.rsi_feature = RSIFeature(period=14)
        self.rsi_period = 14
        self.oversold = 30
        self.overbought = 70
        self.position = 0
    
    def generate_raw_signal(self, data: pl.DataFrame, **kwargs) -> pl.Series:
        """Generate raw trading signals using PositionState enums"""
        
        rsi_values = self.rsi_feature.calculate(data)
        
        # Create signals list for easier manipulation
        signals_list = [PositionState.NEUTRAL] * len(data)
        
        for i in range(1, len(data)):
            current_rsi = rsi_values[i]
            previous_rsi = rsi_values[i-1]
            
            # Skip if RSI values are NaN
            if current_rsi is None or previous_rsi is None or str(current_rsi) == 'nan' or str(previous_rsi) == 'nan':
                continue
            
            # Enter long position: RSI was oversold and now breaks above oversold
            if (previous_rsi <= self.oversold and current_rsi > self.oversold and self.position == 0):
                signals_list[i] = PositionState.LONG
                self.position = 1
            # Exit long position: RSI was overbought and now breaks below overbought
            elif (previous_rsi >= self.overbought and current_rsi < self.overbought and self.position == 1):
                signals_list[i] = PositionState.NEUTRAL
                self.position = 0
            else:
                # Maintain current position
                if self.position == 1:
                    signals_list[i] = PositionState.LONG
                else:
                    signals_list[i] = PositionState.NEUTRAL
        
        # Convert list to Polars Series
        signals = pl.Series(signals_list)
        return signals
