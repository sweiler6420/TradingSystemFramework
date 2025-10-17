"""
Interactive Plot Creator
========================

Creates truly interactive plots using plotly for the research framework.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional
from framework.signals import PositionState, SignalChange


class InteractivePlotCreator:
    """Creates interactive plots for research results"""
    
    def __init__(self, plots_dir: str):
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)
    
    def create_interactive_analysis(self, data: pd.DataFrame, signal_result, 
                                   results: Dict[str, Any], test_name: str = "insample_excellence"):
        """Create interactive HTML analysis plot"""
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.express as px
            
            print(f"\n=== CREATING INTERACTIVE HTML PLOT ===")
            
            # Create subplots with shared x-axis
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Price and Signal Changes', 'Position States', 
                              'Equity Curve Comparison', 'Performance Summary'),
                shared_xaxes=True,  # Link x-axis across all subplots
                vertical_spacing=0.08,
                specs=[[{"secondary_y": False}],
                       [{"secondary_y": False}],
                       [{"secondary_y": False}],
                       [{"secondary_y": False}]]
            )
            
            # 1. Price and Signal Changes
            fig.add_trace(
                go.Scatter(x=data.index, y=data['close'], 
                          mode='lines', name='Price', 
                          line=dict(color='blue', width=1)),
                row=1, col=1
            )
            
            # Add signal markers
            plot_data = signal_result.get_signal_changes_for_plotting()
            if len(plot_data) > 0:
                for _, row in plot_data.iterrows():
                    signal_type = row['signal_change']
                    price = data.loc[row['timestamp'], 'close']
                    
                    # Convert matplotlib markers to plotly symbols
                    plotly_symbol = self._convert_marker_to_plotly(signal_type.plot_marker)
                    
                    fig.add_trace(
                        go.Scatter(x=[row['timestamp']], y=[price],
                                  mode='markers', name=str(signal_type),
                                  marker=dict(
                                      color=signal_type.plot_color,
                                      symbol=plotly_symbol,
                                      size=12
                                  ),
                                  showlegend=True),
                        row=1, col=1
                    )
            
            # 2. Position States
            long_periods = signal_result.position_signals == PositionState.LONG.value
            short_periods = signal_result.position_signals == PositionState.SHORT.value
            neutral_periods = signal_result.position_signals == PositionState.NEUTRAL.value
            
            # Create position visualization
            position_values = np.where(long_periods, 1, 
                                    np.where(short_periods, -1, 0))
            
            fig.add_trace(
                go.Scatter(x=data.index, y=position_values,
                          mode='lines', name='Position',
                          line=dict(color='purple', width=2),
                          fill='tonexty'),
                row=2, col=1
            )
            
            # Add position area fills
            fig.add_trace(
                go.Scatter(x=data.index, y=np.where(long_periods, 1, np.nan),
                          mode='lines', name='Long Position',
                          fill='tozeroy', fillcolor='rgba(0,255,0,0.3)',
                          line=dict(width=0)),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=data.index, y=np.where(short_periods, -1, np.nan),
                          mode='lines', name='Short Position',
                          fill='tozeroy', fillcolor='rgba(255,0,0,0.3)',
                          line=dict(width=0)),
                row=2, col=1
            )
            
            # 3. Equity Curve
            returns = signal_result.position_signals * np.log(data['close']).diff().shift(-1)
            cumulative_returns = (1 + returns).cumprod()
            buy_hold_returns = (1 + np.log(data['close']).diff().shift(-1)).cumprod()
            
            fig.add_trace(
                go.Scatter(x=data.index, y=cumulative_returns,
                          mode='lines', name='Strategy Equity',
                          line=dict(color='blue', width=2)),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=data.index, y=buy_hold_returns,
                          mode='lines', name='Buy & Hold',
                          line=dict(color='gray', width=1)),
                row=3, col=1
            )
            
            # 4. Performance Summary (as text)
            performance_text = f"""
Performance Summary:<br>
Total Return: {results['total_return']:.2%}<br>
Profit Factor: {results['profit_factor']:.2f}<br>
Sharpe Ratio: {results['sharpe_ratio']:.2f}<br>
Sortino Ratio: {results['sortino_ratio']:.2f}<br>
Max Drawdown: {results['max_drawdown']:.2%}<br>
Win Rate: {results['win_rate']:.2%}
"""
            
            fig.add_trace(
                go.Scatter(x=[0], y=[0], mode='text',
                          text=[performance_text],
                          textposition='middle center',
                          showlegend=False),
                row=4, col=1
            )
            
            # Update layout with enhanced x-axis linking
            fig.update_layout(
                title=f'{test_name.replace("_", " ").title()} - Interactive Analysis',
                height=1200,
                showlegend=True,
                hovermode='x unified',  # Unified hover across all subplots
                dragmode='pan',  # Default to pan mode
                xaxis=dict(
                    rangeslider=dict(visible=False),  # Hide range slider for cleaner look
                    type='date'
                )
            )
            
            # Update axes with better x-axis configuration
            fig.update_xaxes(
                title_text="Date", 
                row=4, col=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Position", row=2, col=1)
            fig.update_yaxes(title_text="Cumulative Returns", row=3, col=1)
            fig.update_yaxes(visible=False, row=4, col=1)
            fig.update_xaxes(visible=False, row=4, col=1)
            
            # Save as HTML
            html_file = f"{self.plots_dir}/{test_name}_interactive.html"
            fig.write_html(html_file)
            
            print(f"Interactive HTML plot saved to: {html_file}")
            print("Open this file in a web browser for full interactivity!")
            
            return html_file
            
        except ImportError:
            print("Plotly not available. Install with: pip install plotly")
            print("Falling back to static plots...")
            return None
    
    def create_simple_interactive(self, data: pd.DataFrame, signal_result, 
                                 results: Dict[str, Any], test_name: str = "insample_excellence"):
        """Create a simple interactive plot using matplotlib with interactive backend"""
        
        print(f"\n=== CREATING MATPLOTLIB INTERACTIVE PLOT ===")
        
        import matplotlib.pyplot as plt
        
        # Enable interactive mode
        plt.ion()
        
        # Create the plot
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        
        # 1. Price and Signal Changes
        ax1 = axes[0]
        ax1.plot(data.index, data['close'], label='Price', alpha=0.7, linewidth=1)
        
        # Add signal markers
        plot_data = signal_result.get_signal_changes_for_plotting()
        if len(plot_data) > 0:
            for _, row in plot_data.iterrows():
                signal_type = row['signal_change']
                price = data.loc[row['timestamp'], 'close']
                ax1.scatter(row['timestamp'], price, 
                           color=signal_type.plot_color, 
                           marker=signal_type.plot_marker, 
                           s=100, alpha=0.9, label=str(signal_type), zorder=5)
        
        ax1.set_title('Price and Signal Changes')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Position States
        ax2 = axes[1]
        long_periods = signal_result.position_signals == PositionState.LONG.value
        short_periods = signal_result.position_signals == PositionState.SHORT.value
        neutral_periods = signal_result.position_signals == PositionState.NEUTRAL.value
        
        ax2.fill_between(data.index, 0, 1, where=long_periods, 
                        color='green', alpha=0.3, label='Long Position')
        ax2.fill_between(data.index, -1, 0, where=short_periods, 
                        color='red', alpha=0.3, label='Short Position')
        ax2.fill_between(data.index, -0.5, 0.5, where=neutral_periods, 
                        color='gray', alpha=0.3, label='Neutral Position')
        
        ax2.set_title('Position States')
        ax2.set_ylabel('Position')
        ax2.set_ylim(-1.1, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Equity Curve
        ax3 = axes[2]
        returns = signal_result.position_signals * np.log(data['close']).diff().shift(-1)
        cumulative_returns = (1 + returns).cumprod()
        buy_hold_returns = (1 + np.log(data['close']).diff().shift(-1)).cumprod()
        
        ax3.plot(data.index, cumulative_returns, label='Strategy Equity', 
                color='blue', linewidth=2)
        ax3.plot(data.index, buy_hold_returns, label='Buy & Hold', 
                color='gray', alpha=0.7)
        ax3.set_title('Equity Curve Comparison')
        ax3.set_ylabel('Cumulative Returns')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Summary
        ax4 = axes[3]
        ax4.axis('off')
        performance_text = f"""
Performance Summary:
Total Return: {results['total_return']:.2%}
Profit Factor: {results['profit_factor']:.2f}
Sharpe Ratio: {results['sharpe_ratio']:.2f}
Sortino Ratio: {results['sortino_ratio']:.2f}
Max Drawdown: {results['max_drawdown']:.2%}
Win Rate: {results['win_rate']:.2%}
"""
        ax4.text(0.1, 0.5, performance_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        
        # Save as SVG for better interactivity
        svg_file = f"{self.plots_dir}/{test_name}_interactive.svg"
        plt.savefig(svg_file, bbox_inches='tight')
        
        print(f"Interactive SVG plot saved to: {svg_file}")
        print("This plot will be interactive when displayed!")
        
        # Show the plot (interactive)
        plt.show()
        
        return svg_file
    
    def _convert_marker_to_plotly(self, matplotlib_marker: str) -> str:
        """Convert matplotlib marker symbols to plotly symbols"""
        marker_map = {
            '^': 'triangle-up',      # Up triangle
            'v': 'triangle-down',    # Down triangle  
            'x': 'x',                # X
            'o': 'circle',           # Circle
            's': 'square',           # Square
            'D': 'diamond',          # Diamond
            '+': 'cross',            # Cross
            '*': 'asterisk',         # Asterisk
            '.': 'circle',           # Point
            ',': 'circle',           # Pixel
            '1': 'triangle-up',      # Tri up
            '2': 'triangle-down',    # Tri down
            '3': 'triangle-left',    # Tri left
            '4': 'triangle-right',   # Tri right
            'p': 'pentagon',         # Pentagon
            'h': 'hexagon',          # Hexagon
            'H': 'hexagon',          # Hexagon2
            '8': 'octagon',          # Octagon
        }
        
        return marker_map.get(matplotlib_marker, 'circle')  # Default to circle
