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
    
    def __init__(self, data: pl.DataFrame = None, period: int = 14, overbought: float = 70.0, oversold: float = 30.0):
        """
        Initialize RSI feature.
        
        Args:
            data: DataFrame with OHLCV data
            period: Number of periods for RSI calculation
            overbought: Overbought threshold (typically 70)
            oversold: Oversold threshold (typically 30)
        """
        # Set attributes before calling super().__init__ to avoid calculation issues
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
        super().__init__(name="RSI", data=data, period=period, overbought=overbought, oversold=oversold)
        
    def calculate(self) -> pl.Series:
        """
        Calculate RSI values using stored data.
        
        Returns:
            Series with RSI values (0-100)
        """
        if self.data is None:
            raise ValueError("No data available for RSI calculation")
            
        if not self.validate_data(self.data):
            raise ValueError("Data must contain OHLCV columns")
            
        # Calculate price changes
        price_changes = self.data.select(
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
    
    def get_overbought_signals(self, threshold: Optional[float] = None) -> pl.Series:
        """
        Get overbought signals.
        
        Args:
            threshold: Overbought threshold (uses instance default if None)
            
        Returns:
            Series with boolean overbought signals
        """
        if threshold is None:
            threshold = self.overbought
            
        rsi_values = self.get_values()
        return rsi_values > threshold
    
    def get_oversold_signals(self, threshold: Optional[float] = None) -> pl.Series:
        """
        Get oversold signals.
        
        Args:
            threshold: Oversold threshold (uses instance default if None)
            
        Returns:
            Series with boolean oversold signals
        """
        if threshold is None:
            threshold = self.oversold
            
        rsi_values = self.get_values()
        return rsi_values < threshold
    
    def get_momentum_signals(self, overbought_threshold: Optional[float] = None,
                           oversold_threshold: Optional[float] = None) -> Dict[str, pl.Series]:
        """
        Get both overbought and oversold signals.
        
        Args:
            overbought_threshold: Overbought threshold (uses instance default if None)
            oversold_threshold: Oversold threshold (uses instance default if None)
            
        Returns:
            Dictionary with 'overbought' and 'oversold' signals
        """
        return {
            'overbought': self.get_overbought_signals(overbought_threshold),
            'oversold': self.get_oversold_signals(oversold_threshold)
        }
    
    def get_divergence_signals(self, lookback: int = 5) -> Dict[str, pl.Series]:
        """
        Get divergence signals (price vs RSI).
        
        Args:
            lookback: Number of periods to look back for divergence
            
        Returns:
            Dictionary with 'bullish_divergence' and 'bearish_divergence' signals
        """
        rsi_values = self.get_values()
        
        # Calculate price and RSI trends using rolling windows
        price_trend = self.data.select(
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
    
    def get_rsi_level(self) -> pl.Series:
        """
        Get RSI level classification (oversold, neutral, overbought).
        
        Returns:
            Series with level classification (0=oversold, 1=neutral, 2=overbought)
        """
        rsi_values = self.get_values()
        
        level = pl.Series([1] * len(rsi_values))  # Default to neutral
        level = pl.when(rsi_values < self.oversold).then(0).otherwise(level)
        level = pl.when(rsi_values > self.overbought).then(2).otherwise(level)
        
        return level
    
    def get_normalized_rsi(self) -> pl.Series:
        """
        Get RSI values normalized to 0-1 scale.
        
        Returns:
            Series with normalized RSI values (0-1)
        """
        rsi_values = self.get_values()
        return rsi_values / 100.0
    
    def get_plot(self, x_range=None, **kwargs):
        """
        Generate an RSI plot with overbought/oversold levels.
        
        Args:
            x_range: Optional x-axis range to synchronize with other plots
            **kwargs: Additional plotting parameters
            
        Returns:
            Bokeh figure object for RSI analysis
        """
        if self.data is None:
            raise ValueError("No data available for RSI plotting")
            
        try:
            from bokeh.plotting import figure
            from bokeh.models import Range1d
            
            # Get RSI values (already calculated)
            rsi_values = self.get_values()
            
            # Create RSI plot
            rsi_plot = figure(
                title=f"RSI Analysis (Period: {self.period})",
                x_axis_type='datetime',
                height=300,
                width=1000,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above"
            )
            
            # Synchronize x-axis if provided
            if x_range is not None:
                rsi_plot.x_range = x_range
            
            # Add RSI line
            rsi_plot.line(self.data['timestamp'], rsi_values, 
                         line_color='blue', line_width=2, legend_label='RSI')
            
            # Add overbought/oversold levels
            rsi_plot.line(self.data['timestamp'], [self.overbought] * len(self.data), 
                         line_color='red', line_dash='dashed', 
                         legend_label=f'Overbought ({self.overbought})')
            rsi_plot.line(self.data['timestamp'], [self.oversold] * len(self.data), 
                         line_color='red', line_dash='dashed', 
                         legend_label=f'Oversold ({self.oversold})')
            
            # Set RSI range and styling
            rsi_plot.y_range = Range1d(0, 100)
            rsi_plot.legend.location = "top_left"
            rsi_plot.legend.click_policy = "hide"
            
            return rsi_plot
            
        except ImportError:
            print("Warning: Bokeh not available for RSI plotting")
            return None
