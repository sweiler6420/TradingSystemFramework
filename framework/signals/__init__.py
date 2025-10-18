"""
Signal System Package
====================

A standardized signal system for all trading strategies that provides:
- Explicit position states
- Clear signal changes
- Consistent plotting and analysis
"""

import pandas as pd
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
    position_signals: pd.Series  # Position states (1, -1, 0) for return calculation
    signal_changes: pd.Series    # Signal changes for plotting and analysis
    raw_signals: Optional[pd.Series] = None  # Optional raw signals before position management
    
    def get_position_counts(self) -> Dict[str, int]:
        """Get count of each position state"""
        counts = {}
        for state in PositionState:
            counts[state.name] = (self.position_signals == state.value).sum()
        return counts
    
    def get_signal_change_counts(self) -> Dict[str, int]:
        """Get count of each signal change type"""
        valid_changes = self.signal_changes.dropna()
        counts = {}
        for change in SignalChange:
            count = (valid_changes == change).sum()
            if count > 0:
                counts[change.value] = count
        return counts
    
    def get_signal_changes_for_plotting(self) -> pd.DataFrame:
        """Get signal changes formatted for plotting"""
        valid_changes = self.signal_changes.dropna()
        
        if len(valid_changes) == 0:
            return pd.DataFrame()
        
        plot_data = []
        for timestamp, change in valid_changes.items():
            plot_data.append({
                'timestamp': timestamp,
                'signal_change': change,
                'color': change.plot_color,
                'marker': change.plot_marker
            })
        
        return pd.DataFrame(plot_data)


class SignalManager:
    """Manages signal generation and position state transitions"""
    
    def __init__(self, long_only: bool = False):
        self.long_only = long_only
        self.current_position = PositionState.NEUTRAL
    
    def generate_signals(self, raw_signals: pd.Series, 
                        exit_conditions: Optional[pd.Series] = None) -> SignalResult:
        """Generate position signals and signal changes from SignalChange enums
        
        Args:
            raw_signals: SignalChange enums (NEUTRAL_TO_LONG, LONG_TO_NEUTRAL, etc.)
            exit_conditions: Optional conditions for exiting positions (True = exit)
            
        Returns:
            SignalResult: Contains position signals and signal changes
        """
        
        position_signals = pd.Series(PositionState.NEUTRAL, index=raw_signals.index)
        signal_changes = pd.Series(SignalChange.NO_CHANGE, index=raw_signals.index)
        
        self.current_position = PositionState.NEUTRAL
        
        for i in range(len(raw_signals)):
            signal_change = raw_signals.iloc[i]
            exit_condition = exit_conditions.iloc[i] if exit_conditions is not None else False
            
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
            position_signals.iloc[i] = self.current_position
            signal_changes.iloc[i] = signal_change
        
        return SignalResult(
            position_signals=position_signals,
            signal_changes=signal_changes,
            raw_signals=raw_signals
        )
    
    def _apply_signal_change(self, signal_change: SignalChange) -> PositionState:
        """Apply signal change to determine new position"""
        
        if signal_change == SignalChange.NEUTRAL_TO_LONG:
            return PositionState.LONG
        elif signal_change == SignalChange.NEUTRAL_TO_SHORT:
            if self.long_only:
                return PositionState.NEUTRAL  # Can't go short in long-only mode
            return PositionState.SHORT
        elif signal_change == SignalChange.LONG_TO_NEUTRAL:
            return PositionState.NEUTRAL
        elif signal_change == SignalChange.SHORT_TO_NEUTRAL:
            return PositionState.NEUTRAL
        elif signal_change == SignalChange.LONG_TO_SHORT:
            if self.long_only:
                return PositionState.NEUTRAL  # Can't go short in long-only mode
            return PositionState.SHORT
        elif signal_change == SignalChange.SHORT_TO_LONG:
            return PositionState.LONG
        elif signal_change == SignalChange.NO_CHANGE:
            return self.current_position
        else:
            return self.current_position


def plot_signals(data: pd.DataFrame, signal_result: SignalResult, 
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
    ax.plot(data.index, data['close'], label='Price', alpha=0.7, linewidth=1)
    
    # Get signal changes for plotting
    plot_data = signal_result.get_signal_changes_for_plotting()
    
    if len(plot_data) > 0:
        # Group by signal type for legend
        signal_types = {}
        
        for _, row in plot_data.iterrows():
            signal_type = row['signal_change']
            if signal_type not in signal_types:
                signal_types[signal_type] = {
                    'timestamps': [],
                    'prices': [],
                    'color': signal_type.plot_color,
                    'marker': signal_type.plot_marker
                }
            
            signal_types[signal_type]['timestamps'].append(row['timestamp'])
            signal_types[signal_type]['prices'].append(data.loc[row['timestamp'], 'close'])
        
        # Plot each signal type
        for signal_type, data_dict in signal_types.items():
            ax.scatter(data_dict['timestamps'], data_dict['prices'], 
                      color=data_dict['color'], marker=data_dict['marker'], 
                      s=100, alpha=0.9, label=str(signal_type), zorder=5)
    
    ax.set_title(title)
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_position_states(data: pd.DataFrame, signal_result: SignalResult, 
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
    
    ax.fill_between(data.index, 0, 1, where=long_periods, 
                    color='green', alpha=0.3, label='Long Position')
    ax.fill_between(data.index, -1, 0, where=short_periods, 
                    color='red', alpha=0.3, label='Short Position')
    ax.fill_between(data.index, -0.5, 0.5, where=neutral_periods, 
                    color='gray', alpha=0.3, label='Neutral Position')
    
    ax.set_title(title)
    ax.set_ylabel('Position')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)


def calculate_strategy_returns(data: pd.DataFrame, signal_result: SignalResult) -> pd.Series:
    """Calculate strategy returns using position signals
    
    Args:
        data: Price data with 'close' column
        signal_result: SignalResult from signal generation
        
    Returns:
        pd.Series: Strategy returns
    """
    if 'return' not in data.columns:
        data['return'] = np.log(data['close']).diff().shift(-1)
    
    # Convert PositionState enums to numeric values for return calculation (optimized)
    # Use astype() to avoid pandas warnings
    numeric_signals = signal_result.position_signals.map({
        PositionState.LONG: 1,
        PositionState.SHORT: -1,
        PositionState.NEUTRAL: 0
    }).astype(float)
    
    return numeric_signals * data['return']
