"""
Strategy Optimizers
===================

Optimizers that coordinate feature optimization and strategy-specific parameters.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from framework.strategies.base_strategy import BaseStrategy
from .feature_optimizer import FeatureOptimizer, RSIFeatureOptimizer, DonchianFeatureOptimizer


class StrategyOptimizer(ABC):
    """
    Abstract base class for strategy optimization.
    
    Strategy optimizers coordinate feature optimization and handle
    strategy-specific parameters and feature interactions.
    """
    
    @abstractmethod
    def optimize_strategy(self, data: pd.DataFrame, strategy: BaseStrategy, **kwargs) -> Dict[str, Any]:
        """
        Optimize the complete strategy including all features.
        
        Args:
            data: Market data
            strategy: Strategy to optimize
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary with optimized parameters and performance metrics
        """
        pass


class RSIBreakoutStrategyOptimizer(StrategyOptimizer):
    """
    Optimizer for RSI breakout strategy.
    
    This optimizer:
    1. Uses RSI feature optimizer for RSI parameters
    2. Optimizes any strategy-specific parameters
    3. Coordinates the optimization process
    """
    
    def __init__(self):
        self.rsi_optimizer = RSIFeatureOptimizer()
    
    def optimize_strategy(self, data: pd.DataFrame, strategy: BaseStrategy, **kwargs) -> Dict[str, Any]:
        """
        Optimize RSI breakout strategy.
        
        Args:
            data: Market data
            strategy: RSIBreakoutStrategy instance
            **kwargs: Optimization parameters
        """
        from framework.strategies.rsi_breakout_strategy import RSIBreakoutStrategy
        
        if not isinstance(strategy, RSIBreakoutStrategy):
            raise ValueError("Strategy must be an RSIBreakoutStrategy instance")
        
        # 1. Optimize RSI feature parameters
        rsi_params = self.rsi_optimizer.optimize_feature_params(
            data, strategy.rsi_feature, **kwargs
        )
        
        # 2. Optimize strategy-specific parameters (if any)
        # For now, RSI breakout strategy doesn't have additional parameters
        # but this is where we would add them
        
        # 3. Return optimized parameters
        return {
            'rsi_period': rsi_params['rsi_period'],
            'buy_threshold': rsi_params['buy_threshold'],
            'sell_threshold': rsi_params['sell_threshold'],
            'optimization_results': rsi_params
        }


class DonchianBreakoutStrategyOptimizer(StrategyOptimizer):
    """
    Optimizer for Donchian breakout strategy.
    """
    
    def __init__(self):
        self.donchian_optimizer = DonchianFeatureOptimizer()
    
    def optimize_strategy(self, data: pd.DataFrame, strategy: BaseStrategy, **kwargs) -> Dict[str, Any]:
        """
        Optimize Donchian breakout strategy.
        """
        from framework.strategies.donchian_breakout_strategy import DonchianBreakoutStrategy
        
        if not isinstance(strategy, DonchianBreakoutStrategy):
            raise ValueError("Strategy must be a DonchianBreakoutStrategy instance")
        
        # Optimize Donchian feature parameters
        donchian_params = self.donchian_optimizer.optimize_feature_params(
            data, strategy.donchian_feature, **kwargs
        )
        
        return {
            'lookback': donchian_params['lookback'],
            'optimization_results': donchian_params
        }


class MultiFeatureStrategyOptimizer(StrategyOptimizer):
    """
    Optimizer for strategies using multiple features.
    
    This optimizer can handle strategies that combine multiple features
    and optimize their interactions.
    """
    
    def __init__(self):
        self.feature_optimizers = {
            'rsi': RSIFeatureOptimizer(),
            'donchian': DonchianFeatureOptimizer(),
        }
    
    def optimize_strategy(self, data: pd.DataFrame, strategy: BaseStrategy, **kwargs) -> Dict[str, Any]:
        """
        Optimize multi-feature strategy.
        
        This is a more complex optimization that:
        1. Optimizes each feature independently
        2. Optimizes feature interactions/weights
        3. Tests different feature combinations
        """
        # For now, this is a placeholder for future multi-feature strategies
        # The actual implementation would depend on the specific multi-feature strategy
        
        optimized_params = {}
        
        # Example: Optimize each feature if it exists in the strategy
        if hasattr(strategy, 'rsi_feature'):
            rsi_params = self.feature_optimizers['rsi'].optimize_feature_params(
                data, strategy.rsi_feature, **kwargs
            )
            optimized_params.update(rsi_params)
        
        if hasattr(strategy, 'donchian_feature'):
            donchian_params = self.feature_optimizers['donchian'].optimize_feature_params(
                data, strategy.donchian_feature, **kwargs
            )
            optimized_params.update(donchian_params)
        
        return optimized_params
