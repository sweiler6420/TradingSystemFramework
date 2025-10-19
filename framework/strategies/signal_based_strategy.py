"""
Base Strategy with Signal System
===============================

A base strategy class that uses the standardized signal system.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, Optional, Tuple
from framework.strategies.base_strategy import BaseStrategy
from framework.signals import (
    PositionState, SignalChange, SignalResult, SignalManager,
    plot_signals, plot_position_states, calculate_strategy_returns
)


class SignalBasedStrategy(BaseStrategy):
    """Base strategy class that uses the standardized signal system"""
    
    def __init__(self, name: str, data: pl.DataFrame):
        super().__init__(name, data)
        self.signal_manager = SignalManager()
        
    def generate_signals(self, data: Optional[pl.DataFrame] = None, **kwargs) -> SignalResult:
        """Generate signals using the standardized signal system
        
        This method generates raw signals and processes them through the signal system.
        The base class handles position management and signal changes.
        
        Args:
            data: Price data (optional, uses self.data if not provided)
            **kwargs: Strategy-specific parameters
        
        Returns:
            SignalResult: Contains position signals and signal changes
        """
        # Use provided data or fall back to self.data
        if data is None:
            data = self.data

        # Generate raw signals
        raw_signals = self.generate_raw_signal(**kwargs)
        
        # Auto-detect signal type - check the actual values, not the types
        if isinstance(raw_signals[0], SignalChange):
            signal_type = SignalChange
        elif isinstance(raw_signals[0], PositionState):
            signal_type = PositionState
        else:
            # If it's a string, check if it matches SignalChange values
            first_signal_str = str(raw_signals[0])
            if first_signal_str in [e.value for e in SignalChange]:
                signal_type = SignalChange
            elif first_signal_str in [str(e.value) for e in PositionState]:
                signal_type = PositionState
            else:
                signal_type = SignalChange  # Default to SignalChange
        
        # Generate exit conditions
        exit_conditions = self.generate_exit_conditions(data, **kwargs)
        
        # Use signal manager to generate position signals and changes
        signal_result = self.signal_manager.generate_signals(raw_signals, exit_conditions, signal_type)
        
        return signal_result
    
    def generate_raw_signal(self, **kwargs) -> pl.Series:
        """Generate raw trading signals using SignalChange enums
        
        This method should be implemented by subclasses.
        
        Args:
            **kwargs: Strategy-specific parameters
            
        Returns:
            pl.Series: SignalChange enums (NEUTRAL_TO_LONG, LONG_TO_NEUTRAL, etc.)
        """
        raise NotImplementedError("Subclasses must implement generate_raw_signal")

    
    def generate_exit_conditions(self, data: pl.DataFrame, **kwargs) -> Optional[pl.Series]:
        """Generate exit conditions (True = exit position)
        
        This method can be overridden by subclasses for custom exit logic.
        
        Args:
            data: Price data
            **kwargs: Strategy-specific parameters
            
        Returns:
            Optional[pl.Series]: Exit conditions (True = exit, False/None = hold)
        """
        return None
    
    
    def _calculate_strategy_returns(self, data: pl.DataFrame, signal_result: SignalResult) -> pl.Series:
        """Calculate strategy returns from signal result"""
        return calculate_strategy_returns(data, signal_result)

    
    def plot_strategy_signals(self, data: pl.DataFrame, signal_result: SignalResult, title: Optional[str] = None, ax=None) -> None:
        """Plot strategy signals using the standardized system"""
        if title is None:
            title = f"{self.name} - Signals"
        plot_signals(data, signal_result, title, ax)

    
    def plot_strategy_positions(self, data: pl.DataFrame, signal_result: SignalResult, title: Optional[str] = None, ax=None) -> None:
        """Plot strategy position states using the standardized system"""
        if title is None:
            title = f"{self.name} - Position States"
        plot_position_states(data, signal_result, title, ax)

    
    def get_strategy_summary(self, signal_result: SignalResult) -> Dict[str, Any]:
        """Get a summary of the strategy's signal behavior"""
        return {
            'position_counts': signal_result.get_position_counts(),
            'signal_change_counts': signal_result.get_signal_change_counts(),
            'total_signals': len(signal_result.signal_changes.drop_nulls()),
        }


