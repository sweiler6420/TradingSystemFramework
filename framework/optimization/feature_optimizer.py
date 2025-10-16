"""
Feature Optimizers
==================

Optimizers specifically designed for individual features.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from framework.features.base_feature import BaseFeature


class FeatureOptimizer(ABC):
    """
    Abstract base class for feature optimization.
    
    Feature optimizers focus on optimizing parameters for individual features,
    making them reusable across different strategies.
    """
    
    @abstractmethod
    def optimize_feature_params(self, data: pd.DataFrame, feature: BaseFeature, **kwargs) -> Dict[str, Any]:
        """
        Optimize parameters for a specific feature.
        
        Args:
            data: Market data
            feature: Feature to optimize
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary with optimized parameters and performance metrics
        """
        pass
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor from returns"""
        winning_trades = returns[returns > 0].sum()
        losing_trades = returns[returns < 0].abs().sum()
        return winning_trades / losing_trades if losing_trades > 0 else 0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio from returns"""
        if returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(252)  # Annualized


class RSIFeatureOptimizer(FeatureOptimizer):
    """
    Optimizer specifically for RSI feature parameters.
    
    Optimizes:
    - RSI period
    - Overbought threshold
    - Oversold threshold
    """
    
    def optimize_feature_params(self, data: pd.DataFrame, feature: BaseFeature, **kwargs) -> Dict[str, Any]:
        """
        Optimize RSI parameters.
        
        Args:
            data: Market data
            feature: RSI feature instance
            **kwargs: Optimization parameters
                - rsi_periods: List of periods to test
                - buy_thresholds: List of oversold thresholds
                - sell_thresholds: List of overbought thresholds
                - optimization_metric: 'profit_factor' or 'sharpe_ratio'
        """
        from framework.features.rsi_feature import RSIFeature
        
        if not isinstance(feature, RSIFeature):
            raise ValueError("Feature must be an RSIFeature instance")
        
        # Get parameter ranges
        rsi_periods = kwargs.get('rsi_periods', [10, 14, 21, 30])
        buy_thresholds = kwargs.get('buy_thresholds', [15, 20, 25, 30])
        sell_thresholds = kwargs.get('sell_thresholds', [70, 75, 80, 85])
        optimization_metric = kwargs.get('optimization_metric', 'profit_factor')
        
        best_params = {}
        best_performance = -np.inf
        
        for period in rsi_periods:
            for buy_thresh in buy_thresholds:
                for sell_thresh in sell_thresholds:
                    if buy_thresh >= sell_thresh:
                        continue
                    
                    # Create temporary feature with these parameters
                    temp_feature = RSIFeature(
                        period=period,
                        oversold=buy_thresh,
                        overbought=sell_thresh
                    )
                    
                    # Calculate feature signals
                    momentum_signals = temp_feature.get_momentum_signals(data)
                    
                    # Generate simple trading signals (buy oversold, sell overbought)
                    signals = pd.Series(0, index=data.index)
                    for i in range(len(data)):
                        if momentum_signals['oversold'].iloc[i]:
                            signals.iloc[i] = 1  # Buy
                        elif momentum_signals['overbought'].iloc[i]:
                            signals.iloc[i] = 0  # Sell
                    
                    # Calculate returns
                    if 'return' not in data.columns:
                        returns = np.log(data['close']).diff().shift(-1)
                    else:
                        returns = data['return']
                    
                    strategy_returns = signals * returns
                    
                    # Calculate performance metric
                    if optimization_metric == 'profit_factor':
                        performance = self._calculate_profit_factor(strategy_returns)
                    elif optimization_metric == 'sharpe_ratio':
                        performance = self._calculate_sharpe_ratio(strategy_returns)
                    else:
                        raise ValueError(f"Unknown optimization metric: {optimization_metric}")
                    
                    if performance > best_performance:
                        best_performance = performance
                        best_params = {
                            'rsi_period': period,
                            'buy_threshold': buy_thresh,
                            'sell_threshold': sell_thresh,
                            f'best_{optimization_metric}': performance
                        }
        
        return best_params


class DonchianFeatureOptimizer(FeatureOptimizer):
    """
    Optimizer specifically for Donchian feature parameters.
    
    Optimizes:
    - Lookback period
    - Breakout thresholds
    """
    
    def optimize_feature_params(self, data: pd.DataFrame, feature: BaseFeature, **kwargs) -> Dict[str, Any]:
        """
        Optimize Donchian parameters.
        
        Args:
            data: Market data
            feature: Donchian feature instance
            **kwargs: Optimization parameters
                - lookback_periods: List of lookback periods to test
                - upper_thresholds: List of upper breakout thresholds
                - lower_thresholds: List of lower breakdown thresholds
                - optimization_metric: 'profit_factor' or 'sharpe_ratio'
        """
        from framework.features.donchian_feature import DonchianFeature
        
        if not isinstance(feature, DonchianFeature):
            raise ValueError("Feature must be a DonchianFeature instance")
        
        # Get parameter ranges
        lookback_periods = kwargs.get('lookback_periods', [15, 20, 25, 30, 40])
        upper_thresholds = kwargs.get('upper_thresholds', [1.0, 1.01, 1.02])
        lower_thresholds = kwargs.get('lower_thresholds', [0.98, 0.99, 1.0])
        optimization_metric = kwargs.get('optimization_metric', 'profit_factor')
        
        best_params = {}
        best_performance = -np.inf
        
        for lookback in lookback_periods:
            for upper_thresh in upper_thresholds:
                for lower_thresh in lower_thresholds:
                    if lower_thresh >= upper_thresh:
                        continue
                    
                    # Create temporary feature with these parameters
                    temp_feature = DonchianFeature(lookback=lookback)
                    
                    # Get breakout signals
                    breakout_signals = temp_feature.get_breakout_signals(
                        data, upper_thresh, lower_thresh
                    )
                    
                    # Generate trading signals
                    signals = pd.Series(0, index=data.index)
                    signals[breakout_signals['upper_breakout']] = 1   # Long
                    signals[breakout_signals['lower_breakdown']] = -1  # Short
                    
                    # Forward fill signals
                    signals = signals.replace(0, np.nan).ffill().fillna(0)
                    
                    # Calculate returns
                    if 'return' not in data.columns:
                        returns = np.log(data['close']).diff().shift(-1)
                    else:
                        returns = data['return']
                    
                    strategy_returns = signals * returns
                    
                    # Calculate performance metric
                    if optimization_metric == 'profit_factor':
                        performance = self._calculate_profit_factor(strategy_returns)
                    elif optimization_metric == 'sharpe_ratio':
                        performance = self._calculate_sharpe_ratio(strategy_returns)
                    else:
                        raise ValueError(f"Unknown optimization metric: {optimization_metric}")
                    
                    if performance > best_performance:
                        best_performance = performance
                        best_params = {
                            'lookback': lookback,
                            'upper_threshold': upper_thresh,
                            'lower_threshold': lower_thresh,
                            f'best_{optimization_metric}': performance
                        }
        
        return best_params
