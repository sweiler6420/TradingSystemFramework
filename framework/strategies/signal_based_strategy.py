"""
Base Strategy with Signal System
===============================

A base strategy class that uses the standardized signal system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from framework.strategies.base_strategy import BaseStrategy, Optimizer
from framework.signals import (
    PositionState, SignalChange, SignalResult, SignalManager,
    plot_signals, plot_position_states, calculate_strategy_returns
)


class SignalBasedStrategy(BaseStrategy):
    """Base strategy class that uses the standardized signal system"""
    
    def __init__(self, name: str, long_only: bool = False):
        super().__init__(name)
        self.signal_manager = SignalManager(long_only=long_only)
        self.long_only = long_only
        
    def generate_signals(self, **kwargs) -> SignalResult:
        """Generate signals using the standardized signal system
        
        This method should be implemented by subclasses to generate raw signals.
        The base class handles position management and signal changes.
        
        Returns:
            SignalResult: Contains position signals and signal changes
        """
        return self._generate_signals_internal(self.data, **kwargs)
    
    def generate_raw_signals(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate raw trading signals (1, -1, 0)
        
        This method should be implemented by subclasses.
        
        Args:
            data: Price data
            **kwargs: Strategy-specific parameters
            
        Returns:
            pd.Series: Raw signals (1 for buy, -1 for sell, 0 for hold)
        """
        raise NotImplementedError("Subclasses must implement generate_raw_signals")
    
    def generate_exit_conditions(self, data: pd.DataFrame, **kwargs) -> Optional[pd.Series]:
        """Generate exit conditions (True = exit position)
        
        This method can be overridden by subclasses for custom exit logic.
        
        Args:
            data: Price data
            **kwargs: Strategy-specific parameters
            
        Returns:
            Optional[pd.Series]: Exit conditions (True = exit, False/None = hold)
        """
        return None
    
    def _generate_signals_internal(self, data: pd.DataFrame, **kwargs) -> SignalResult:
        """Internal method to generate signals using the signal system"""
        
        # Generate raw signals
        raw_signals = self.generate_raw_signals(data, **kwargs)
        
        # Generate exit conditions
        exit_conditions = self.generate_exit_conditions(data, **kwargs)
        
        # Use signal manager to generate position signals and changes
        signal_result = self.signal_manager.generate_signals(raw_signals, exit_conditions)
        
        return signal_result
    
    def _calculate_strategy_returns(self, data: pd.DataFrame, signal_result: SignalResult) -> pd.Series:
        """Calculate strategy returns from signal result"""
        return calculate_strategy_returns(data, signal_result)
    
    def plot_strategy_signals(self, data: pd.DataFrame, signal_result: SignalResult, 
                             title: Optional[str] = None, ax=None) -> None:
        """Plot strategy signals using the standardized system"""
        if title is None:
            title = f"{self.name} - Signals"
        plot_signals(data, signal_result, title, ax)
    
    def plot_strategy_positions(self, data: pd.DataFrame, signal_result: SignalResult, 
                               title: Optional[str] = None, ax=None) -> None:
        """Plot strategy position states using the standardized system"""
        if title is None:
            title = f"{self.name} - Position States"
        plot_position_states(data, signal_result, title, ax)
    
    def get_strategy_summary(self, signal_result: SignalResult) -> Dict[str, Any]:
        """Get a summary of the strategy's signal behavior"""
        return {
            'position_counts': signal_result.get_position_counts(),
            'signal_change_counts': signal_result.get_signal_change_counts(),
            'total_signals': len(signal_result.signal_changes.dropna()),
            'strategy_type': 'Long-Only' if self.long_only else 'Long-Short'
        }


class SignalBasedOptimizer(Optimizer):
    """Base optimizer class for signal-based strategies"""
    
    def optimize(self, data: pd.DataFrame, strategy: SignalBasedStrategy, **kwargs) -> Dict[str, Any]:
        """Optimize signal-based strategy parameters"""
        
        best_params = {}
        best_performance = -np.inf
        
        # Get parameter ranges from kwargs
        param_ranges = self._get_parameter_ranges(**kwargs)
        
        # Test all parameter combinations
        for params in self._generate_parameter_combinations(param_ranges):
            
            # Generate signals with these parameters
            signal_result = strategy._generate_signals_internal(data, **params)
            
            # Calculate performance
            returns = strategy._calculate_strategy_returns(data, signal_result)
            performance = self._calculate_performance_metric(returns)
            
            if performance > best_performance:
                best_performance = performance
                best_params = params.copy()
                best_params['performance'] = performance
        
        return best_params
    
    def _get_parameter_ranges(self, **kwargs) -> Dict[str, list]:
        """Get parameter ranges from kwargs - override in subclasses"""
        return {}
    
    def _generate_parameter_combinations(self, param_ranges: Dict[str, list]) -> list:
        """Generate all parameter combinations"""
        import itertools
        
        if not param_ranges:
            return [{}]
        
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _calculate_performance_metric(self, returns: pd.Series) -> float:
        """Calculate performance metric - override in subclasses"""
        # Default: profit factor
        winning_trades = returns[returns > 0].sum()
        losing_trades = returns[returns < 0].abs().sum()
        return winning_trades / losing_trades if losing_trades > 0 else 0
