"""
Donchian Channel Feature
========================

Donchian Channel is a trend-following indicator that shows the highest high
and lowest low over a specified lookback period.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from framework.features.base_feature import BaseFeature


class DonchianFeature(BaseFeature):
    """
    Donchian Channel feature.
    
    The Donchian Channel consists of:
    - Upper Band: Highest high over the lookback period
    - Lower Band: Lowest low over the lookback period
    - Middle Band: Average of upper and lower bands
    
    This feature can be used for:
    - Breakout strategies
    - Trend identification
    - Support/resistance levels
    """
    
    def __init__(self, lookback: int = 20, include_middle: bool = True):
        """
        Initialize Donchian feature.
        
        Args:
            lookback: Number of periods to look back for highs/lows
            include_middle: Whether to include middle band (average of upper/lower)
        """
        super().__init__(
            name="Donchian",
            lookback=lookback,
            include_middle=include_middle
        )
        self.lookback = lookback
        self.include_middle = include_middle
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Donchian Channel values.
        
        Returns the middle band by default. Use get_bands() for all bands.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with Donchian middle band values
        """
        if not self.validate_data(data):
            raise ValueError("Data must contain OHLCV columns")
            
        # Calculate Donchian bands
        upper = data['high'].rolling(window=self.lookback).max()
        lower = data['low'].rolling(window=self.lookback).min()
        middle = (upper + lower) / 2
        
        if self.include_middle:
            return middle
        else:
            return upper  # Default to upper band if middle not included
    
    def get_bands(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get all Donchian bands.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with 'upper', 'lower', and 'middle' bands
        """
        if not self.validate_data(data):
            raise ValueError("Data must contain OHLCV columns")
            
        upper = data['high'].rolling(window=self.lookback).max()
        lower = data['low'].rolling(window=self.lookback).min()
        middle = (upper + lower) / 2
        
        return {
            'upper': upper,
            'lower': lower,
            'middle': middle
        }
    
    def get_breakout_signals(self, data: pd.DataFrame, 
                           upper_threshold: float = 1.0,
                           lower_threshold: float = 1.0) -> Dict[str, pd.Series]:
        """
        Get breakout signals based on Donchian channels.
        
        Args:
            data: DataFrame with OHLCV data
            upper_threshold: Multiplier for upper breakout (e.g., 1.0 = exact breakout)
            lower_threshold: Multiplier for lower breakdown (e.g., 1.0 = exact breakdown)
            
        Returns:
            Dictionary with 'upper_breakout' and 'lower_breakdown' signals
        """
        bands = self.get_bands(data)
        
        upper_breakout = data['close'] > (bands['upper'] * upper_threshold)
        lower_breakdown = data['close'] < (bands['lower'] * lower_threshold)
        
        return {
            'upper_breakout': upper_breakout,
            'lower_breakdown': lower_breakdown
        }
    
    def get_channel_width(self, data: pd.DataFrame) -> pd.Series:
        """
        Get the width of the Donchian channel.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with channel width (upper - lower)
        """
        bands = self.get_bands(data)
        return bands['upper'] - bands['lower']
    
    def get_channel_position(self, data: pd.DataFrame) -> pd.Series:
        """
        Get the position of close price within the channel (0-1 scale).
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with position (0 = lower band, 1 = upper band)
        """
        bands = self.get_bands(data)
        channel_width = bands['upper'] - bands['lower']
        
        # Avoid division by zero
        position = (data['close'] - bands['lower']) / channel_width.replace(0, np.nan)
        
        return position
