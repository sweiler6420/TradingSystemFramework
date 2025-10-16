"""
RSI Strategy Implementation
==========================
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any
from framework.strategies.base_strategy import BaseStrategy, Optimizer


class RSIOptimizer(Optimizer):
    """Optimizer for RSI strategy parameters"""
    
    def optimize(self, data: pd.DataFrame, strategy, **kwargs) -> Dict[str, Any]:
        """Optimize RSI period and thresholds"""
        
        best_params = {}
        best_performance = -np.inf
        
        # Parameter ranges to test
        rsi_periods = kwargs.get('rsi_periods', [10, 14, 21, 30])
        buy_thresholds = kwargs.get('buy_thresholds', [10, 15, 20, 25, 30])
        sell_thresholds = kwargs.get('sell_thresholds', [70, 75, 80, 85, 90])
        
        for period in rsi_periods:
            for buy_thresh in buy_thresholds:
                for sell_thresh in sell_thresholds:
                    if buy_thresh >= sell_thresh:
                        continue
                        
                    # Generate signals with these parameters
                    signals = strategy._generate_rsi_signals(
                        data, period, buy_thresh, sell_thresh
                    )
                    
                    # Calculate performance
                    returns = strategy._calculate_strategy_returns(data, signals)
                    profit_factor = self._calculate_profit_factor(returns)
                    
                    if profit_factor > best_performance:
                        best_performance = profit_factor
                        best_params = {
                            'rsi_period': period,
                            'buy_threshold': buy_thresh,
                            'sell_threshold': sell_thresh,
                            'profit_factor': profit_factor
                        }
        
        return best_params
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor"""
        winning_trades = returns[returns > 0].sum()
        losing_trades = returns[returns < 0].abs().sum()
        return winning_trades / losing_trades if losing_trades > 0 else 0


class RSIStrategy(BaseStrategy):
    """RSI-based long-only trading strategy"""
    
    def __init__(self, rsi_period: int = 14, buy_threshold: float = 20, sell_threshold: float = 80):
        super().__init__("RSI Strategy")
        self.rsi_period = rsi_period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        

    def generate_signals(self, **kwargs) -> pd.Series:
        """Generate RSI-based trading signals"""
        
        # Override parameters if provided
        rsi_period = kwargs.get('rsi_period', self.rsi_period)
        buy_threshold = kwargs.get('buy_threshold', self.buy_threshold)
        sell_threshold = kwargs.get('sell_threshold', self.sell_threshold)
        
        return self._generate_rsi_signals(self.data, rsi_period, buy_threshold, sell_threshold)
    

    def _generate_rsi_signals(self, data: pd.DataFrame, rsi_period: int, buy_threshold: float, sell_threshold: float) -> pd.Series:
        """Internal method to generate RSI signals"""
        
        # Calculate RSI
        rsi = ta.rsi(data['close'], length=rsi_period)
        
        # Generate signals (long-only)
        signal = pd.Series(0, index=data.index)
        in_position = False
        
        for i in range(len(data)):
            rsi_value = rsi.iloc[i]
            
            if not in_position and rsi_value < buy_threshold:
                # Enter long position when oversold
                signal.iloc[i] = 1
                in_position = True
            elif in_position and rsi_value > sell_threshold:
                # Exit long position when overbought
                signal.iloc[i] = 0
                in_position = False
            elif in_position:
                # Hold long position
                signal.iloc[i] = 1
            else:
                # No position
                signal.iloc[i] = 0
        
        return signal
    

    def _calculate_strategy_returns(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Calculate strategy returns from signals"""
        if 'return' not in data.columns:
            data['return'] = np.log(data['close']).diff().shift(-1)
        return signals * data['return']
