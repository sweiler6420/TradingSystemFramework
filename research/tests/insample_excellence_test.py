"""
In-Sample Excellence Test Implementation
======================================

A standardized test for proof-of-concept validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Tuple
import os

# Framework imports
from framework import (
    DataHandler, SignalBasedStrategy, SignalBasedOptimizer,
    RSIFeature, DonchianFeature, PositionState, SignalChange
)
from framework.performance import (
    ProfitFactorMeasure, SharpeRatioMeasure, SortinoRatioMeasure,
    MaxDrawdownMeasure, TotalReturnMeasure, WinRateMeasure
)


class InSampleExcellenceTest:
    """Standardized in-sample excellence test for proof of concept validation"""
    
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.results_dir = os.path.join(project_dir, 'results')
        self.plots_dir = os.path.join(project_dir, 'plots')
        
        # Initialize performance measures
        self.measures = {
            'profit_factor': ProfitFactorMeasure(),
            'sharpe_ratio': SharpeRatioMeasure(),
            'sortino_ratio': SortinoRatioMeasure(),
            'max_drawdown': MaxDrawdownMeasure(),
            'total_return': TotalReturnMeasure(),
            'win_rate': WinRateMeasure()
        }
    
    def run_test(self, strategy: SignalBasedStrategy, data_handler: DataHandler, 
                 test_name: str = "insample_excellence") -> Dict[str, Any]:
        """
        Run the in-sample excellence test
        
        Args:
            strategy: Strategy to test
            data_handler: Data handler with loaded data
            test_name: Name for saving results
            
        Returns:
            Dictionary with test results and metadata
        """
        print(f"=== {test_name.upper().replace('_', ' ')} TEST ===")
        print(f"Strategy: {strategy.name}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get data
        data = data_handler.get_data()
        print(f"Data loaded: {data.shape[0]} rows from {data.index[0]} to {data.index[-1]}")
        print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        # Set strategy data
        strategy.set_data_handler(data_handler)
        
        # Generate signals
        signal_result = strategy.generate_signals()
        
        # Get strategy summary
        summary = strategy.get_strategy_summary(signal_result)
        print(f"\nStrategy Type: {summary['strategy_type']}")
        print(f"Position Counts: {summary['position_counts']}")
        print(f"Signal Changes: {summary['signal_change_counts']}")
        print(f"Total Signals: {summary['total_signals']}")
        
        # Calculate performance
        returns = strategy._calculate_strategy_returns(data, signal_result)
        
        # Calculate all metrics
        results = {}
        for name, measure in self.measures.items():
            results[name] = measure.calculate(returns)
        
        # Print results
        print(f"\n=== PERFORMANCE RESULTS ===")
        for name, value in results.items():
            print(f"{name.title()}: {value:.4f}")
        
        # Create test metadata
        test_metadata = {
            'test_name': test_name,
            'strategy_name': strategy.name,
            'test_date': datetime.now().isoformat(),
            'data_period': f"{data.index[0]} to {data.index[-1]}",
            'data_points': len(data),
            'strategy_summary': summary,
            'performance_results': results
        }
        
        # Save results
        self._save_results(test_metadata, test_name)
        
        print(f"\nResults saved to: {self.results_dir}/{test_name}_results.csv")
        print(f"Metadata saved to: {self.results_dir}/{test_name}_metadata.json")
        
        return test_metadata
    
    def create_performance_plots(self, data: pd.DataFrame, signal_result, 
                                results: Dict[str, Any], test_name: str = "insample_excellence"):
        """Create comprehensive performance visualization plots"""
        print(f"\n=== CREATING PERFORMANCE PLOTS ===")
        
        # Create static plots (multiple formats)
        self._create_static_plots(data, signal_result, results, test_name)
        
        # Try to create interactive plots
        try:
            from .interactive_plot_creator import InteractivePlotCreator
            interactive_creator = InteractivePlotCreator(self.plots_dir)
            
            # Try HTML interactive plot first
            html_file = interactive_creator.create_interactive_analysis(data, signal_result, results, test_name)
            
            if html_file is None:
                # Fall back to matplotlib interactive
                interactive_creator.create_simple_interactive(data, signal_result, results, test_name)
                
        except ImportError:
            print("Interactive plot creator not available. Using static plots only.")
    
    def _create_static_plots(self, data: pd.DataFrame, signal_result, 
                            results: Dict[str, Any], test_name: str = "insample_excellence"):
        """Create static plots in multiple formats"""
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        
        # 1. Price and Signal Changes
        ax1 = axes[0]
        # Plot price
        ax1.plot(data.index, data['close'], label='Price', alpha=0.7, linewidth=1)
        
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
                ax1.scatter(data_dict['timestamps'], data_dict['prices'], 
                          color=data_dict['color'], marker=data_dict['marker'], 
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
        
        # Save in multiple formats
        base_name = f"{self.plots_dir}/{test_name}_analysis"
        
        # Ensure plots directory exists
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # # Save as PNG (high quality for documents)
        # plt.savefig(f'{base_name}.png', dpi=300, bbox_inches='tight')
        
        # # Save as SVG (vector format, more interactive)
        # plt.savefig(f'{base_name}.svg', bbox_inches='tight')
        
        # # Save as PDF (vector format, good for publications)
        # plt.savefig(f'{base_name}.pdf', bbox_inches='tight')
        
        plt.show()
        
        print(f"Static plots saved to:")
        print(f"  - {base_name}.png (high quality)")
        print(f"  - {base_name}.svg (vector, interactive)")
        print(f"  - {base_name}.pdf (vector, publication)")
    
    def _save_results(self, test_metadata: Dict[str, Any], test_name: str):
        """Save test results to files"""
        import json
        
        # Ensure directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Save performance results as CSV
        results_df = pd.DataFrame([test_metadata['performance_results']])
        results_file = os.path.join(self.results_dir, f"{test_name}_results.csv")
        results_df.to_csv(results_file, index=False)
        
        # Save metadata as JSON
        metadata_file = os.path.join(self.results_dir, f"{test_name}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(test_metadata, f, indent=2, default=str)
    
    def generate_test_report(self, test_metadata: Dict[str, Any], test_name: str = "insample_excellence"):
        """Generate a comprehensive test report"""
        report_content = f"""# {test_name.replace('_', ' ').title()} Test Report

**Test Date:** {test_metadata['test_date']}
**Strategy:** {test_metadata['strategy_name']}
**Data Period:** {test_metadata['data_period']}
**Data Points:** {test_metadata['data_points']}

## Performance Results

| Metric | Value |
|--------|-------|
| Total Return | {test_metadata['performance_results']['total_return']:.2%} |
| Profit Factor | {test_metadata['performance_results']['profit_factor']:.2f} |
| Sharpe Ratio | {test_metadata['performance_results']['sharpe_ratio']:.2f} |
| Sortino Ratio | {test_metadata['performance_results']['sortino_ratio']:.2f} |
| Max Drawdown | {test_metadata['performance_results']['max_drawdown']:.2%} |
| Win Rate | {test_metadata['performance_results']['win_rate']:.2%} |

## Strategy Summary

- **Strategy Type:** {test_metadata['strategy_summary']['strategy_type']}
- **Total Signals:** {test_metadata['strategy_summary']['total_signals']}
- **Position Counts:** {test_metadata['strategy_summary']['position_counts']}

## Signal Changes

{test_metadata['strategy_summary']['signal_change_counts']}

## Analysis

*Add your analysis and observations here...*

## Next Steps

*Add next steps and recommendations here...*
"""
        
        report_file = os.path.join(self.results_dir, f"{test_name}_report.md")
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"Test report saved to: {report_file}")
