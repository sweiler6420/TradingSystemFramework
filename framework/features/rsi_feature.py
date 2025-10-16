"""
RSI Feature
===========

Relative Strength Index (RSI) is a momentum oscillator that measures the speed
and magnitude of price changes.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, Optional
from framework.features.base_feature import BaseFeature


class RSIFeature(BaseFeature):
    """
    Relative Strength Index (RSI) feature.
    
    RSI oscillates between 0 and 100 and is typically used to identify:
    - Overbought conditions (RSI > 70)
    - Oversold conditions (RSI < 30)
    - Divergence patterns
    - Momentum shifts
    
    This feature can be used for:
    - Mean reversion strategies
    - Momentum strategies
    - Signal confirmation
    - Risk management
    """
    
    def __init__(self, period: int = 14, 
                 overbought: float = 70.0,
                 oversold: float = 30.0):
        """
        Initialize RSI feature.
        
        Args:
            period: Number of periods for RSI calculation
            overbought: Overbought threshold (typically 70)
            oversold: Oversold threshold (typically 30)
        """
        super().__init__(
            name="RSI",
            period=period,
            overbought=overbought,
            oversold=oversold
        )
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate RSI values.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with RSI values (0-100)
        """
        if not self.validate_data(data):
            raise ValueError("Data must contain OHLCV columns")
            
        return ta.rsi(data['close'], length=self.period)
    
    def get_overbought_signals(self, data: pd.DataFrame, 
                              threshold: Optional[float] = None) -> pd.Series:
        """
        Get overbought signals.
        
        Args:
            data: DataFrame with OHLCV data
            threshold: Overbought threshold (uses instance default if None)
            
        Returns:
            Series with boolean overbought signals
        """
        if threshold is None:
            threshold = self.overbought
            
        rsi_values = self.calculate(data)
        return rsi_values > threshold
    
    def get_oversold_signals(self, data: pd.DataFrame,
                            threshold: Optional[float] = None) -> pd.Series:
        """
        Get oversold signals.
        
        Args:
            data: DataFrame with OHLCV data
            threshold: Oversold threshold (uses instance default if None)
            
        Returns:
            Series with boolean oversold signals
        """
        if threshold is None:
            threshold = self.oversold
            
        rsi_values = self.calculate(data)
        return rsi_values < threshold
    
    def get_momentum_signals(self, data: pd.DataFrame,
                           overbought_threshold: Optional[float] = None,
                           oversold_threshold: Optional[float] = None) -> Dict[str, pd.Series]:
        """
        Get both overbought and oversold signals.
        
        Args:
            data: DataFrame with OHLCV data
            overbought_threshold: Overbought threshold (uses instance default if None)
            oversold_threshold: Oversold threshold (uses instance default if None)
            
        Returns:
            Dictionary with 'overbought' and 'oversold' signals
        """
        return {
            'overbought': self.get_overbought_signals(data, overbought_threshold),
            'oversold': self.get_oversold_signals(data, oversold_threshold)
        }
    
    def get_divergence_signals(self, data: pd.DataFrame, 
                             lookback: int = 5) -> Dict[str, pd.Series]:
        """
        Get divergence signals (price vs RSI).
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of periods to look back for divergence
            
        Returns:
            Dictionary with 'bullish_divergence' and 'bearish_divergence' signals
        """
        rsi_values = self.calculate(data)
        
        # Calculate price and RSI trends
        price_trend = data['close'].rolling(window=lookback).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False
        )
        rsi_trend = rsi_values.rolling(window=lookback).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False
        )
        
        # Divergence signals
        bullish_divergence = (price_trend == -1) & (rsi_trend == 1)  # Price down, RSI up
        bearish_divergence = (price_trend == 1) & (rsi_trend == -1)  # Price up, RSI down
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }
    
    def get_rsi_level(self, data: pd.DataFrame) -> pd.Series:
        """
        Get RSI level classification (oversold, neutral, overbought).
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with level classification (0=oversold, 1=neutral, 2=overbought)
        """
        rsi_values = self.calculate(data)
        
        level = pd.Series(1, index=data.index)  # Default to neutral
        level[rsi_values < self.oversold] = 0  # Oversold
        level[rsi_values > self.overbought] = 2  # Overbought
        
        return level
    
    def get_normalized_rsi(self, data: pd.DataFrame) -> pd.Series:
        """
        Get RSI values normalized to 0-1 scale.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with normalized RSI values (0-1)
        """
        rsi_values = self.calculate(data)
        return rsi_values / 100.0
