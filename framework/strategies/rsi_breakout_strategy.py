"""
RSI Strategy Implementation
==========================
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from framework.strategies.base_strategy import BaseStrategy, Optimizer
from framework.features.rsi_feature import RSIFeature


class RSIBreakoutOptimizer(Optimizer):
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
                    signals = strategy._generate_rsi_signals_with_params(
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


class RSIBreakoutStrategy(BaseStrategy):
    """RSI-based long-only trading strategy using RSIFeature"""
    
    def __init__(self, rsi_period: int = 14, buy_threshold: float = 20, sell_threshold: float = 80):
        super().__init__("RSI Strategy")
        self.rsi_feature = RSIFeature(
            period=rsi_period,
            oversold=buy_threshold,
            overbought=sell_threshold
        )
        

    def generate_signals(self, **kwargs) -> pd.Series:
        """Generate RSI-based trading signals"""
        
        # Override parameters if provided
        rsi_period = kwargs.get('rsi_period', self.rsi_feature.period)
        buy_threshold = kwargs.get('buy_threshold', self.rsi_feature.oversold)
        sell_threshold = kwargs.get('sell_threshold', self.rsi_feature.overbought)
        
        # Update feature parameters if changed
        if (rsi_period != self.rsi_feature.period or 
            buy_threshold != self.rsi_feature.oversold or 
            sell_threshold != self.rsi_feature.overbought):
            self.rsi_feature.set_params(
                period=rsi_period,
                oversold=buy_threshold,
                overbought=sell_threshold
            )
        
        return self._generate_rsi_signals(self.data)
    

    def _generate_rsi_signals(self, data: pd.DataFrame) -> pd.Series:
        """Internal method to generate RSI signals using RSIFeature"""
        
        # Get momentum signals from the feature
        momentum_signals = self.rsi_feature.get_momentum_signals(data)
        
        # Generate long-only trading signals
        signal = pd.Series(0, index=data.index)
        in_position = False
        
        for i in range(len(data)):
            is_oversold = momentum_signals['oversold'].iloc[i]
            is_overbought = momentum_signals['overbought'].iloc[i]
            
            if not in_position and is_oversold:
                # Enter long position when oversold
                signal.iloc[i] = 1
                in_position = True
            elif in_position and is_overbought:
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
    
    def _generate_rsi_signals_with_params(self, data: pd.DataFrame, 
                                        rsi_period: int, 
                                        buy_threshold: float, 
                                        sell_threshold: float) -> pd.Series:
        """Internal method to generate RSI signals with specific parameters for optimization"""
        
        # Create temporary RSI feature with specific parameters
        temp_rsi_feature = RSIFeature(
            period=rsi_period,
            oversold=buy_threshold,
            overbought=sell_threshold
        )
        
        # Get momentum signals from the temporary feature
        momentum_signals = temp_rsi_feature.get_momentum_signals(data)
        
        # Generate long-only trading signals
        signal = pd.Series(0, index=data.index)
        in_position = False
        
        for i in range(len(data)):
            is_oversold = momentum_signals['oversold'].iloc[i]
            is_overbought = momentum_signals['overbought'].iloc[i]
            
            if not in_position and is_oversold:
                # Enter long position when oversold
                signal.iloc[i] = 1
                in_position = True
            elif in_position and is_overbought:
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
