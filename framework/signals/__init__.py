"""
Signal System Package
====================

A standardized signal system for all trading strategies that provides:
- Explicit position states
- Clear signal changes
- Consistent plotting and analysis
"""

import polars as pl
import numpy as np
from enum import Enum
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


class PositionState(Enum):
    """Position states for trading strategies"""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0
    
    def __str__(self):
        return self.name
    
    @property
    def value_for_returns(self) -> int:
        """Get the numeric value for return calculations"""
        return self.value


class SignalChange(Enum):
    """Signal changes for plotting and analysis"""
    # Position changes
    NEUTRAL_TO_LONG = "NEUTRAL_TO_LONG"      # Enter long from cash
    NEUTRAL_TO_SHORT = "NEUTRAL_TO_SHORT"    # Enter short from cash
    LONG_TO_NEUTRAL = "LONG_TO_NEUTRAL"      # Exit long to cash
    LONG_TO_SHORT = "LONG_TO_SHORT"          # Switch from long to short
    SHORT_TO_NEUTRAL = "SHORT_TO_NEUTRAL"    # Exit short to cash
    SHORT_TO_LONG = "SHORT_TO_LONG"          # Switch from short to long
    NO_CHANGE = "NO_CHANGE"                  # No position change   
    
    
    def __str__(self):
        return self.value
    
    @property
    def is_entry(self) -> bool:
        """Check if this is an entry signal"""
        return "TO_LONG" in self.value or "TO_SHORT" in self.value
    
    @property
    def is_exit(self) -> bool:
        """Check if this is an exit signal"""
        return "TO_NEUTRAL" in self.value
    
    @property
    def is_long_signal(self) -> bool:
        """Check if this involves a long position"""
        return "LONG" in self.value
    
    @property
    def is_short_signal(self) -> bool:
        """Check if this involves a short position"""
        return "SHORT" in self.value
    
    @property
    def plot_color(self) -> str:
        """Get the color for plotting this signal"""
        if self.is_entry and self.is_long_signal:
            return 'green'
        elif self.is_entry and self.is_short_signal:
            return 'red'
        elif self.is_exit:
            return 'orange'
        else:
            return 'blue'
    
    @property
    def plot_marker(self) -> str:
        """Get the marker for plotting this signal"""
        if self.is_entry and self.is_long_signal:
            return '^'
        elif self.is_entry and self.is_short_signal:
            return 'v'
        elif self.is_exit:
            return 'x'
        else:
            return 'o'


@dataclass
class SignalResult:
    """Result from signal generation containing position states and changes"""
    position_signals: pl.Series  # Position states (1, -1, 0) for return calculation
    signal_changes: pl.Series    # Signal changes for plotting and analysis
    raw_signals: Optional[pl.Series] = None  # Optional raw signals before position management
    
    def get_position_counts(self) -> Dict[str, int]:
        """Get count of each position state"""
        counts = {}
        for state in PositionState:
            counts[state.name] = (self.position_signals == state.value).sum()
        return counts
    
    def get_signal_change_counts(self) -> Dict[str, int]:
        """Get count of each signal change type"""
        valid_changes = self.signal_changes.drop_nulls()
        counts = {}
        for change in SignalChange:
            count = (valid_changes == change.value).sum()
            if count > 0:
                counts[change.value] = count
        return counts
    
    def get_signal_changes_for_plotting(self, data: pl.DataFrame = None) -> pl.DataFrame:
        """Get signal changes formatted for plotting
        
        Args:
            data: DataFrame with timestamp column to get actual timestamps
        """
        # Only get actual signal changes, not NO_CHANGE
        signal_changes = self.signal_changes.drop_nulls()
        
        if len(signal_changes) == 0:
            return pl.DataFrame()
        
        plot_data = []
        for i, change_str in enumerate(signal_changes):
            # Skip NO_CHANGE signals
            if change_str == "NO_CHANGE":
                continue
                
            # Convert string back to SignalChange enum
            try:
                change_enum = SignalChange(change_str)
                
                # Get actual timestamp if data is provided
                if data is not None and 'timestamp' in data.columns and i < len(data):
                    actual_timestamp = data['timestamp'][i]
                    actual_price = data['close'][i]
                else:
                    actual_timestamp = i  # Fallback to index
                    actual_price = 0
                
                plot_data.append({
                    'index': i,
                    'timestamp': actual_timestamp,
                    'price': actual_price,
                    'signal_change': change_enum,
                    'color': change_enum.plot_color,
                    'marker': change_enum.plot_marker
                })
            except ValueError:
                # Skip invalid signal changes
                continue
        
        return pl.DataFrame(plot_data)


class SignalManager:
    """Manages signal generation and position state transitions"""
    
    def __init__(self):
        self.current_position = PositionState.NEUTRAL
    
    def generate_signals(self, raw_signals: pl.Series, exit_conditions: Optional[pl.Series] = None, signal_type: Optional[type] = None) -> SignalResult:
        """Generate position signals and signal changes from SignalChange enums or PositionState enums
        
        Args:
            raw_signals: SignalChange enums (NEUTRAL_TO_LONG, LONG_TO_NEUTRAL, etc.) or PositionState enums (LONG, SHORT, NEUTRAL)
            exit_conditions: Optional conditions for exiting positions (True = exit)
            signal_type: Type of signals (SignalChange or PositionState). If None, auto-detect from first element
            
        Returns:
            SignalResult: Contains position signals and signal changes
        """
        
        # Convert to lists for easier manipulation
        raw_signals_list = raw_signals.to_list()
        exit_conditions_list = exit_conditions.to_list() if exit_conditions is not None else [False] * len(raw_signals)
        
        # Auto-detect signal type if not provided
        if signal_type is None and len(raw_signals_list) > 0:
            if isinstance(raw_signals_list[0], SignalChange):
                signal_type = SignalChange
            elif isinstance(raw_signals_list[0], PositionState):
                signal_type = PositionState
            elif isinstance(raw_signals_list[0], str):
                # Convert strings back to SignalChange enums
                signal_type = SignalChange
                raw_signals_list = [self._string_to_signal_change(s) for s in raw_signals_list]
            else:
                raise ValueError(f"Unknown signal type: {type(raw_signals_list[0])}")
        
        position_signals_list = []
        signal_changes_list = []
        
        self.current_position = PositionState.NEUTRAL
        
        for i in range(len(raw_signals_list)):
            raw_signal = raw_signals_list[i]
            exit_condition = exit_conditions_list[i]
            
            # Convert PositionState to SignalChange if needed
            if signal_type == PositionState:
                # raw_signal is a string representation of PositionState
                position_state = PositionState(raw_signal)
                signal_change = self._position_state_to_signal_change(position_state)
            else:
                signal_change = raw_signal
            
            # Handle exit conditions first
            if exit_condition and self.current_position != PositionState.NEUTRAL:
                if self.current_position == PositionState.LONG:
                    signal_change = SignalChange.LONG_TO_NEUTRAL
                elif self.current_position == PositionState.SHORT:
                    signal_change = SignalChange.SHORT_TO_NEUTRAL
            
            # Apply signal change to determine new position
            new_position = self._apply_signal_change(signal_change)
            
            # Update position
            self.current_position = new_position
            position_signals_list.append(self.current_position.value)
            signal_changes_list.append(signal_change.value if hasattr(signal_change, 'value') else str(signal_change))
        
        # Convert back to polars Series
        position_signals = pl.Series(position_signals_list)
        signal_changes = pl.Series(signal_changes_list)
        
        return SignalResult(
            position_signals=position_signals,
            signal_changes=signal_changes,
            raw_signals=raw_signals
        )
    
    def _position_state_to_signal_change(self, position_state: PositionState) -> SignalChange:
        """Convert PositionState to SignalChange based on current position"""
        
        if position_state == PositionState.NEUTRAL:
            if self.current_position == PositionState.LONG:
                return SignalChange.LONG_TO_NEUTRAL
            elif self.current_position == PositionState.SHORT:
                return SignalChange.SHORT_TO_NEUTRAL
            else:
                return SignalChange.NO_CHANGE
                
        elif position_state == PositionState.LONG:
            if self.current_position == PositionState.NEUTRAL:
                return SignalChange.NEUTRAL_TO_LONG
            elif self.current_position == PositionState.SHORT:
                return SignalChange.SHORT_TO_LONG
            else:
                return SignalChange.NO_CHANGE
                
        elif position_state == PositionState.SHORT:
            if self.current_position == PositionState.NEUTRAL:
                return SignalChange.NEUTRAL_TO_SHORT
            elif self.current_position == PositionState.LONG:
                return SignalChange.LONG_TO_SHORT
            else:
                return SignalChange.NO_CHANGE
        
        return SignalChange.NO_CHANGE
    
    def _string_to_signal_change(self, signal_str: str) -> SignalChange:
        """Convert string to SignalChange enum"""
        try:
            return SignalChange(signal_str)
        except ValueError:
            # If it's not a valid SignalChange, return NO_CHANGE
            return SignalChange.NO_CHANGE
    
    def _apply_signal_change(self, signal_change) -> PositionState:
        """Apply signal change to determine new position"""
        
        # Handle both string and enum inputs
        if isinstance(signal_change, str):
            signal_str = signal_change
        elif isinstance(signal_change, SignalChange):
            signal_str = signal_change.value
        else:
            # Try to convert to string
            signal_str = str(signal_change)
        
        if signal_str == "NEUTRAL_TO_LONG":
            return PositionState.LONG
        elif signal_str == "NEUTRAL_TO_SHORT":
            return PositionState.SHORT
        elif signal_str == "LONG_TO_NEUTRAL":
            return PositionState.NEUTRAL
        elif signal_str == "SHORT_TO_NEUTRAL":
            return PositionState.NEUTRAL
        elif signal_str == "LONG_TO_SHORT":
            return PositionState.SHORT
        elif signal_str == "SHORT_TO_LONG":
            return PositionState.LONG
        elif signal_str == "NO_CHANGE":
            return self.current_position
        else:
            return self.current_position


def plot_signals(data: pl.DataFrame, signal_result: SignalResult, 
                title: str = "Strategy Signals", ax=None) -> None:
    """Plot price with signal changes using the standardized signal system
    
    Args:
        data: Price data with 'close' column
        signal_result: SignalResult from signal generation
        title: Title for the plot
        ax: Optional matplotlib axis to plot on
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot price
    ax.plot(range(len(data)), data['close'], label='Price', alpha=0.7, linewidth=1)
    
    # Get signal changes for plotting
    plot_data = signal_result.get_signal_changes_for_plotting()
    
    if len(plot_data) > 0:
        # Group by signal type for legend
        signal_types = {}
        
        for row in plot_data.iter_rows(named=True):
            signal_type = row['signal_change']
            if signal_type not in signal_types:
                signal_types[signal_type] = {
                    'indices': [],
                    'prices': [],
                    'color': signal_type.plot_color,
                    'marker': signal_type.plot_marker
                }
            
            signal_types[signal_type]['indices'].append(row['index'])
            signal_types[signal_type]['prices'].append(data[row['index'], 'close'])
        
        # Plot each signal type
        for signal_type, data_dict in signal_types.items():
            ax.scatter(data_dict['indices'], data_dict['prices'], 
                      color=data_dict['color'], marker=data_dict['marker'], 
                      s=100, alpha=0.9, label=str(signal_type), zorder=5)
    
    ax.set_title(title)
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_position_states(data: pl.DataFrame, signal_result: SignalResult, 
                        title: str = "Position States", ax=None) -> None:
    """Plot position states over time
    
    Args:
        data: Price data (for index alignment)
        signal_result: SignalResult from signal generation
        title: Title for the plot
        ax: Optional matplotlib axis to plot on
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    
    # Create position visualization
    long_periods = signal_result.position_signals == PositionState.LONG.value
    short_periods = signal_result.position_signals == PositionState.SHORT.value
    neutral_periods = signal_result.position_signals == PositionState.NEUTRAL.value
    
    ax.fill_between(range(len(data)), 0, 1, where=long_periods, 
                    color='green', alpha=0.3, label='Long Position')
    ax.fill_between(range(len(data)), -1, 0, where=short_periods, 
                    color='red', alpha=0.3, label='Short Position')
    ax.fill_between(range(len(data)), -0.5, 0.5, where=neutral_periods, 
                    color='gray', alpha=0.3, label='Neutral Position')
    
    ax.set_title(title)
    ax.set_ylabel('Position')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)


def calculate_strategy_returns(data: pl.DataFrame, signal_result: SignalResult) -> pl.Series:
    """Calculate strategy returns using position signals
    
    Args:
        data: Price data with 'close' column
        signal_result: SignalResult from signal generation
        
    Returns:
        pl.Series: Strategy returns
    """
    if 'return' not in data.columns:
        data = data.with_columns(
            pl.col('close').log().diff().shift(-1).alias('return')
        )
    
    # Convert PositionState enums to numeric values for return calculation
    numeric_signals = signal_result.position_signals.cast(pl.Float64)
    
    return numeric_signals * data['return']