"""
Bokeh Interactive Plot Creator
==============================

Creates truly interactive plots using Bokeh, which is specifically designed
for web-based interactive visualizations with excellent built-in saving.
"""

import os
import numpy as np
import polars as pl
from typing import Dict, Any, Optional
from framework.signals import PositionState, SignalChange

# Bokeh imports
from bokeh.plotting import figure, save, output_file
from bokeh.layouts import column
from bokeh.models import HoverTool, CrosshairTool, Range1d
from bokeh.palettes import Category10
from bokeh.io import curdoc, show
from bokeh.resources import CDN


class BokehInteractivePlotCreator:
    """Creates interactive plots using Bokeh"""
    
    def __init__(self, plots_dir: str):
        self.plots_dir = plots_dir
        
    def create_interactive_analysis(self, data: pl.DataFrame, signal_result, 
                                   results: Dict[str, Any], test_name: str = "analysis", 
                                   show_plot: bool = True, version_manager=None, custom_plots=None) -> Optional[str]:
        """Create interactive analysis using Bokeh"""
        
        try:
            print(f"\n=== CREATING BOKEH INTERACTIVE PLOT ===")
            
            # Prepare data
            data_copy = data.clone()
            # Add timestamp column if not present
            if 'timestamp' not in data_copy.columns:
                data_copy = data_copy.with_row_index('row_index')
                data_copy = data_copy.with_columns(
                    pl.col('row_index').cast(pl.Datetime).alias('timestamp')
                )
            
            # Create output file with versioning
            if version_manager:
                versioned_name = version_manager.get_versioned_filename(test_name, "html", "V")
                html_file = f"{self.plots_dir}/{versioned_name}"
            else:
                html_file = f"{self.plots_dir}/{test_name}_bokeh_interactive.html"
            
            output_file(html_file, title=f"{test_name.replace('_', ' ').title()} - Interactive Analysis")
            
            # Color palette
            colors = Category10[10]
            
            # 1. Price and Signals Plot
            p1 = figure(
                title="Price and Signal Changes",
                x_axis_type='datetime',
                width=1000, height=300,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above"
            )
            
            # Price line
            p1.line(data_copy['timestamp'], data_copy['close'], 
                   line_width=2, color=colors[0], legend_label="Price")
            
            # Signal markers (optimized with ColumnDataSource)
            plot_data = signal_result.get_signal_changes_for_plotting(data_copy)
            if len(plot_data) > 0:
                # Convert to lists for filtering
                timestamps = plot_data['timestamp'].to_list()
                prices = plot_data['price'].to_list()
                signal_changes = plot_data['signal_change'].to_list()
                
                # Separate signals by type
                entry_timestamps = []
                entry_prices = []
                exit_timestamps = []
                exit_prices = []
                
                for i, signal in enumerate(signal_changes):
                    signal_str = str(signal)
                    if 'TO_LONG' in signal_str or 'TO_SHORT' in signal_str:
                        entry_timestamps.append(timestamps[i])
                        entry_prices.append(prices[i])
                    elif 'TO_NEUTRAL' in signal_str:
                        exit_timestamps.append(timestamps[i])
                        exit_prices.append(prices[i])
                
                # Plot entry signals (up arrows)
                if entry_timestamps:
                    p1.scatter(entry_timestamps, entry_prices, 
                             size=15, color='green', marker='triangle', 
                             legend_label="Entry Signals", alpha=0.8)
                
                # Plot exit signals (down arrows)  
                if exit_timestamps:
                    p1.scatter(exit_timestamps, exit_prices, 
                             size=15, color='orange', marker='inverted_triangle', 
                             legend_label="Exit Signals", alpha=0.8)
            
            p1.legend.location = "top_left"
            p1.legend.click_policy = "hide"
            p1.yaxis.axis_label = "Price ($)"
            
            # 2. Position States Plot
            p2 = figure(
                title="Position States",
                x_axis_type='datetime',
                width=1000, height=200,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above",
                x_range=p1.x_range  # Link x-axis
            )
            
            # Position areas (optimized, no warnings)
            # Convert to numeric values once for efficiency
            numeric_positions = signal_result.position_signals.cast(pl.Float64)
            
            p2.line(data_copy['timestamp'], numeric_positions, 
                   line_width=2, color=colors[1], legend_label="Position")
            
            # Add area fills (optimized)
            long_mask = numeric_positions == 1
            short_mask = numeric_positions == -1
            
            p2.varea(data_copy['timestamp'], 0.5, 
                    np.where(long_mask, 1, np.nan),
                    color='green', alpha=0.3, legend_label="Long")
            
            p2.varea(data_copy['timestamp'], -0.5, 
                    np.where(short_mask, -1, np.nan),
                    color='red', alpha=0.3, legend_label="Short")
            
            p2.varea(data_copy['timestamp'], -0.5, 0.5, 
                    color='gray', alpha=0.3, legend_label="Neutral")
            
            p2.legend.location = "top_left"
            p2.legend.click_policy = "hide"
            p2.yaxis.axis_label = "Position"
            p2.y_range = Range1d(-1.5, 1.5)
            
            # 3. Equity Curve Plot
            p3 = figure(
                title="Equity Curve Comparison",
                x_axis_type='datetime',
                width=1000, height=200,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above",
                x_range=p1.x_range  # Link x-axis
            )
            
            # Calculate returns (optimized conversion, no warnings)
            numeric_signals = signal_result.position_signals.cast(pl.Float64)
            returns = numeric_signals * np.log(data_copy['close']).diff().shift(-1)
            cumulative_returns = (1 + returns).cum_prod()
            buy_hold_returns = (1 + np.log(data_copy['close']).diff().shift(-1)).cum_prod()
            
            p3.line(data_copy['timestamp'], cumulative_returns, 
                   line_width=2, color=colors[2], legend_label="Strategy")
            p3.line(data_copy['timestamp'], buy_hold_returns, 
                   line_width=2, color=colors[3], legend_label="Buy & Hold")
            
            p3.legend.location = "top_left"
            p3.legend.click_policy = "hide"
            p3.yaxis.axis_label = "Cumulative Returns"
            
            # 4. Performance Summary Plot
            p4 = figure(
                title="Performance Summary",
                x_axis_type='datetime',
                width=1000, height=150,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above",
                x_range=p1.x_range  # Link x-axis
            )
            
            # Performance metrics as text
            pf = results.get('profit_factor', 0)
            sharpe = results.get('sharpe_ratio', 0)
            sortino = results.get('sortino_ratio', 0)
            max_dd = results.get('max_drawdown', 0)
            total_return = results.get('total_return', 0)
            win_rate = results.get('win_rate', 0)
            
            # Add performance text
            p4.text(x=data_copy['timestamp'][0], y=0.5,
                   text=[f"PF: {pf:.4f} | Sharpe: {sharpe:.4f} | Sortino: {sortino:.4f}"],
                   text_font_size="12pt", text_color="black")
            
            p4.text(x=data_copy['timestamp'][0], y=0.3,
                   text=[f"Max DD: {max_dd:.4f} | Total Return: {total_return:.4f} | Win Rate: {win_rate:.4f}"],
                   text_font_size="12pt", text_color="black")
            
            p4.y_range = Range1d(0, 1)
            p4.yaxis.visible = False
            p4.xaxis.visible = False
            
            # Add hover tools
            hover1 = HoverTool(tooltips=[("Date", "@x{%F}"), ("Price", "@y{$0,0.00}")])
            hover1.formatters = {"@x": "datetime"}
            p1.add_tools(hover1)
            
            hover2 = HoverTool(tooltips=[("Date", "@x{%F}"), ("Position", "@y")])
            hover2.formatters = {"@x": "datetime"}
            p2.add_tools(hover2)
            
            hover3 = HoverTool(tooltips=[("Date", "@x{%F}"), ("Returns", "@y{0.0000}")])
            hover3.formatters = {"@x": "datetime"}
            p3.add_tools(hover3)
            
            # Combine plots - insert custom plots between position states (p2) and equity curve (p3)
            layout_plots = [p1, p2]  # Price and position states
            
            # Add custom plots if provided - synchronize x-axis with main price plot
            if custom_plots:
                for custom_plot in custom_plots:
                    # Synchronize x-axis with main price plot for zoom/pan coordination
                    custom_plot.x_range = p1.x_range
                layout_plots.extend(custom_plots)
            
            layout_plots.extend([p3, p4])  # Equity curve and drawdown
            
            layout = column(*layout_plots, sizing_mode="scale_width")
            
            # Save the plot
            save(layout)
            
            # Show plot during runtime if requested
            if show_plot:
                try:
                    show(layout)
                    print("Interactive plot displayed in browser!")
                except Exception as e:
                    print(f"Could not display plot: {e}")
                    print("Plot saved to file instead.")
            
            print(f"Bokeh interactive plot saved to: {html_file}")
            print("Open this file in a web browser for full interactivity!")
            
            return html_file
            
        except ImportError:
            print("Bokeh not available. Install with: pip install bokeh")
            return None
        except Exception as e:
            print(f"Error creating Bokeh plot: {e}")
            return None
    
