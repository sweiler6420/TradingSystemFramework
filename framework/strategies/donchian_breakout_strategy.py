"""
Donchian Breakout Strategy Implementation
========================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from framework.strategies.base_strategy import BaseStrategy, Optimizer
from framework.features.donchian_feature import DonchianFeature


class DonchianBreakoutOptimizer(Optimizer):
    """Optimizer for Donchian breakout strategy parameters"""
    
    def optimize(self, data: pd.DataFrame, strategy, **kwargs) -> Dict[str, Any]:
        """Optimize Donchian breakout lookback period"""
        
        best_params = {}
        best_performance = -np.inf
        
        # Parameter range to test
        lookback_range = kwargs.get('lookback_range', range(12, 169))
        
        for lookback in lookback_range:
            # Generate signals with this lookback
            signals = strategy._generate_donchian_signals_with_params(data, lookback)
            
            # Calculate performance
            returns = strategy._calculate_strategy_returns(data, signals)
            profit_factor = self._calculate_profit_factor(returns)
            
            if profit_factor > best_performance:
                best_performance = profit_factor
                best_params = {
                    'lookback': lookback,
                    'profit_factor': profit_factor
                }
        
        return best_params
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor"""
        winning_trades = returns[returns > 0].sum()
        losing_trades = returns[returns < 0].abs().sum()
        return winning_trades / losing_trades if losing_trades > 0 else 0


class DonchianBreakoutStrategy(BaseStrategy):
    """Donchian breakout strategy using DonchianFeature"""
    
    def __init__(self, lookback: int = 20):
        super().__init__("Donchian Breakout Strategy")
        self.donchian_feature = DonchianFeature(lookback=lookback)
        
    def generate_signals(self, **kwargs) -> pd.Series:
        """Generate Donchian breakout signals"""
        
        # Override parameters if provided
        lookback = kwargs.get('lookback', self.donchian_feature.lookback)
        
        # Update feature parameters if changed
        if lookback != self.donchian_feature.lookback:
            self.donchian_feature.set_params(lookback=lookback)
        
        return self._generate_donchian_signals(self.data)
    
    def _generate_donchian_signals(self, data: pd.DataFrame) -> pd.Series:
        """Internal method to generate Donchian signals using DonchianFeature"""
        
        # Get breakout signals from the feature
        breakout_signals = self.donchian_feature.get_breakout_signals(data)
        
        # Generate trading signals
        signal = pd.Series(0, index=data.index)
        signal[breakout_signals['upper_breakout']] = 1   # Long on upper breakout
        signal[breakout_signals['lower_breakdown']] = -1  # Short on lower breakdown
        
        # Forward fill to maintain position
        signal = signal.replace(0, np.nan).ffill().fillna(0)
        
        return signal
    
    def _generate_donchian_signals_with_params(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        """Internal method to generate Donchian signals with specific parameters for optimization"""
        
        # Create temporary Donchian feature with specific parameters
        temp_donchian_feature = DonchianFeature(lookback=lookback)
        
        # Get breakout signals from the temporary feature
        breakout_signals = temp_donchian_feature.get_breakout_signals(data)
        
        # Generate trading signals
        signal = pd.Series(0, index=data.index)
        signal[breakout_signals['upper_breakout']] = 1   # Long on upper breakout
        signal[breakout_signals['lower_breakdown']] = -1  # Short on lower breakdown
        
        # Forward fill to maintain position
        signal = signal.replace(0, np.nan).ffill().fillna(0)
        
        return signal
    
    def _calculate_strategy_returns(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Calculate strategy returns from signals"""
        if 'return' not in data.columns:
            data['return'] = np.log(data['close']).diff().shift(-1)
        return signals * data['return']
