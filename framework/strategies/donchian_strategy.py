"""
Donchian Strategy Implementation
===============================
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from framework.strategies.base_strategy import BaseStrategy, Optimizer


class DonchianOptimizer(Optimizer):
    """Optimizer for Donchian strategy parameters"""
    
    def optimize(self, data: pd.DataFrame, strategy, **kwargs) -> Dict[str, Any]:
        """Optimize Donchian lookback period"""
        
        best_params = {}
        best_performance = -np.inf
        
        # Parameter range to test
        lookback_range = kwargs.get('lookback_range', range(12, 169))
        
        for lookback in lookback_range:
            # Generate signals with this lookback
            signals = strategy._generate_donchian_signals(data, lookback)
            
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


class DonchianStrategy(BaseStrategy):
    """Donchian breakout strategy"""
    
    def __init__(self, lookback: int = 20):
        super().__init__("Donchian Strategy")
        self.lookback = lookback
        
    def generate_signals(self, **kwargs) -> pd.Series:
        """Generate Donchian breakout signals"""
        
        # Override parameters if provided
        lookback = kwargs.get('lookback', self.lookback)
        
        return self._generate_donchian_signals(self.data, lookback)
    
    def _generate_donchian_signals(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        """Internal method to generate Donchian signals"""
        
        # Calculate Donchian channels
        upper = data['close'].rolling(lookback - 1).max().shift(1)
        lower = data['close'].rolling(lookback - 1).min().shift(1)
        
        # Generate signals
        signal = pd.Series(np.full(len(data), np.nan), index=data.index)
        signal.loc[data['close'] > upper] = 1  # Long on breakout above upper channel
        signal.loc[data['close'] < lower] = -1  # Short on breakdown below lower channel
        
        # Forward fill to maintain position
        signal = signal.ffill()
        
        # Fill initial NaN values with 0 (no position)
        signal = signal.fillna(0)
        
        return signal
    
    def _calculate_strategy_returns(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Calculate strategy returns from signals"""
        if 'return' not in data.columns:
            data['return'] = np.log(data['close']).diff().shift(-1)
        return signals * data['return']
