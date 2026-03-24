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
    PositionState, SignalChange, SignalManager, SignalResult,
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
        
        # Generate exit conditions (may depend on raw_signals, e.g. stop / RR)
        exit_conditions = self.generate_exit_conditions(data, raw_signals=raw_signals, **kwargs)
        
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

    
    def add_price_overlays(
        self,
        price_figure,
        data: pl.DataFrame,
        signal_result: SignalResult,
        *,
        results: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Optional hook: draw on the main Bokeh price figure (same y-scale as OHLC).

        Invoked after candlesticks, close line, and entry/exit markers. Override to add
        EMAs, bands, or horizontal levels. Oscillators (MACD, RSI) usually belong in
        :meth:`create_custom_plots` as separate panes.
        """
        pass

    def get_trade_levels_for_plot(
        self,
        data: pl.DataFrame,
        signal_result: SignalResult,
    ) -> Optional[tuple[pl.Series, pl.Series]]:
        """Optional per-bar stop / take-profit (same length as ``data``) for Bokeh reports.

        Return ``(stop_loss, take_profit)`` with null/NaN when flat. When provided, the
        interactive price pane also draws RR profit/risk rectangles (entry→exit) and
        enriches hovers. Default: ``None`` (strategies with RR levels can override).
        """
        return None

    def generate_exit_conditions(
        self,
        data: pl.DataFrame,
        raw_signals: Optional[pl.Series] = None,
        **kwargs,
    ) -> Optional[pl.Series]:
        """Generate exit conditions (True = exit position)

        This method can be overridden by subclasses for custom exit logic (stops, RR targets,
        trend filters). ``raw_signals`` is the output of :meth:`generate_raw_signal` for the
        same run so exits can reference entries (e.g. per-trade stop from entry bar).

        Args:
            data: Price data
            raw_signals: Raw :class:`SignalChange` series from :meth:`generate_raw_signal`
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


