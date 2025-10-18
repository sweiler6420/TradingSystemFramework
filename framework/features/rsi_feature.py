"""
RSI Feature
===========

Relative Strength Index (RSI) is a momentum oscillator that measures the speed
and magnitude of price changes.
"""

import polars as pl
import numpy as np
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
        
    def calculate(self, data: pl.DataFrame) -> pl.Series:
        """
        Calculate RSI values.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with RSI values (0-100)
        """
        if not self.validate_data(data):
            raise ValueError("Data must contain OHLCV columns")
            
        # Calculate price changes
        price_changes = data.select(
            pl.col('close').diff().alias('price_change')
        )
        
        # Separate gains and losses
        gains_losses = price_changes.select([
            pl.when(pl.col('price_change') > 0)
            .then(pl.col('price_change'))
            .otherwise(0)
            .alias('gains'),
            pl.when(pl.col('price_change') < 0)
            .then(pl.col('price_change').abs())
            .otherwise(0)
            .alias('losses')
        ])
        
        # Calculate smoothed averages using exponential moving average
        alpha = 1.0 / self.period
        rsi_data = gains_losses.with_columns([
            pl.col('gains').ewm_mean(alpha=alpha, adjust=False).alias('avg_gains'),
            pl.col('losses').ewm_mean(alpha=alpha, adjust=False).alias('avg_losses')
        ])
        
        # Calculate RSI
        rsi_values = rsi_data.select(
            pl.when(pl.col('avg_losses') == 0)
            .then(100.0)
            .otherwise(
                100.0 - (100.0 / (1.0 + pl.col('avg_gains') / pl.col('avg_losses')))
            )
            .alias('rsi')
        )
        
        return rsi_values['rsi']
    
    def get_overbought_signals(self, data: pl.DataFrame, 
                              threshold: Optional[float] = None) -> pl.Series:
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
    
    def get_oversold_signals(self, data: pl.DataFrame,
                            threshold: Optional[float] = None) -> pl.Series:
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
    
    def get_momentum_signals(self, data: pl.DataFrame,
                           overbought_threshold: Optional[float] = None,
                           oversold_threshold: Optional[float] = None) -> Dict[str, pl.Series]:
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
    
    def get_divergence_signals(self, data: pl.DataFrame, 
                             lookback: int = 5) -> Dict[str, pl.Series]:
        """
        Get divergence signals (price vs RSI).
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of periods to look back for divergence
            
        Returns:
            Dictionary with 'bullish_divergence' and 'bearish_divergence' signals
        """
        rsi_values = self.calculate(data)
        
        # Calculate price and RSI trends using rolling windows
        price_trend = data.select(
            pl.col('close').rolling_mean(window_size=lookback).alias('price_trend')
        )
        rsi_trend = rsi_values.rolling_mean(window_size=lookback)
        
        # Calculate divergence signals
        bullish_divergence = (price_trend['price_trend'].shift(1) > price_trend['price_trend']) & \
                           (rsi_trend.shift(1) < rsi_trend)
        bearish_divergence = (price_trend['price_trend'].shift(1) < price_trend['price_trend']) & \
                           (rsi_trend.shift(1) > rsi_trend)
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }
    
    def get_rsi_level(self, data: pl.DataFrame) -> pl.Series:
        """
        Get RSI level classification (oversold, neutral, overbought).
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with level classification (0=oversold, 1=neutral, 2=overbought)
        """
        rsi_values = self.calculate(data)
        
        level = pl.Series([1] * len(rsi_values))  # Default to neutral
        level = pl.when(rsi_values < self.oversold).then(0).otherwise(level)
        level = pl.when(rsi_values > self.overbought).then(2).otherwise(level)
        
        return level
    
    def get_normalized_rsi(self, data: pl.DataFrame) -> pl.Series:
        """
        Get RSI values normalized to 0-1 scale.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with normalized RSI values (0-1)
        """
        rsi_values = self.calculate(data)
        return rsi_values / 100.0
