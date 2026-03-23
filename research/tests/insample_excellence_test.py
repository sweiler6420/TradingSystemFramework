"""
In-Sample Excellence Test Implementation
======================================

A standardized test for proof-of-concept validation using Bokeh for interactive plots.
"""

import polars as pl
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple
import os
import textwrap

# Framework imports
from framework import (
    DataHandler, SignalBasedStrategy,
    RSIFeature, DonchianFeature, PositionState, SignalChange
)
from framework.performance import (
    ProfitFactorMeasure, SharpeRatioMeasure, SortinoRatioMeasure,
    MaxDrawdownMeasure, TotalReturnMeasure, WinRateMeasure
)

# Version management
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from research.version_manager import VersionManager

# Bokeh plotting imports
from .bokeh_interactive_plot_creator import BokehInteractivePlotCreator


class InSampleExcellenceTest:
    """Standardized in-sample excellence test for proof of concept validation"""
    
    def __init__(self, project_dir: str, strategy: SignalBasedStrategy):
        self.project_dir = project_dir
        self.results_dir = os.path.join(project_dir, 'results')
        self.plots_dir = os.path.join(project_dir, 'plots')
        self.strategy = strategy
        
        # Initialize performance measures
        self.measures = {
            'profit_factor': ProfitFactorMeasure(),
            'sharpe_ratio': SharpeRatioMeasure(),
            'sortino_ratio': SortinoRatioMeasure(),
            'max_drawdown': MaxDrawdownMeasure(),
            'total_return': TotalReturnMeasure(),
            'win_rate': WinRateMeasure()
        }
        
        # Initialize version manager for both results and plots
        self.version_manager = VersionManager(self.results_dir)
        self.plots_version_manager = VersionManager(self.plots_dir)
    
    def run_test(self, data_handler: DataHandler, 
                 test_name: str = "insample_excellence") -> Dict[str, Any]:
        """
        Run the in-sample excellence test
        
        Args:
            data_handler: Data handler with loaded data
            test_name: Name for saving results
            
        Returns:
            Dictionary with test results and metadata
        """
        print(f"=== {test_name.upper().replace('_', ' ')} TEST ===")
        print(f"Strategy: {self.strategy.name}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get data
        data = data_handler.get_data()
        print(f"Data loaded: {data.shape[0]} rows from {data['timestamp'][0]} to {data['timestamp'][-1]}")
        print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        # Set strategy data
        self.strategy.set_data_handler(data_handler)
        
        # Generate signals
        signal_result = self.strategy.generate_signals()
        
        # Get strategy summary
        summary = self.strategy.get_strategy_summary(signal_result)
        print(f"Position Counts: {summary['position_counts']}")
        print(f"Signal Changes: {summary['signal_change_counts']}")
        print(f"Total Signals: {summary['total_signals']}")
        
        # Calculate performance
        returns = self.strategy._calculate_strategy_returns(data, signal_result)
        
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
            'strategy_name': self.strategy.name,
            'test_date': datetime.now().isoformat(),
            'data_period': f"{data['timestamp'][0]} to {data['timestamp'][-1]}",
            'data_points': len(data),
            'strategy_summary': summary,
            'performance_results': results
        }
        
        # Save results
        self._save_results(test_metadata, test_name)
        
        print(f"\nResults saved to: {self.results_dir}/{test_name}_results.csv")
        print(f"Metadata saved to: {self.results_dir}/{test_name}_metadata.json")
        
        return test_metadata
    
    def create_performance_plots(self, data: pl.DataFrame, signal_result, 
                                results: Dict[str, Any], test_name: str = "insample_excellence", 
                                show_plot: bool = True):
        """Create comprehensive performance visualization plots using Bokeh"""
        print(f"\n=== CREATING PERFORMANCE PLOTS ===")
        
        # Get custom plots from strategy
        custom_plots = []
        if hasattr(self.strategy, 'create_custom_plots'):
            try:
                custom_plots = self.strategy.create_custom_plots(data, signal_result, results=results)
                if custom_plots:
                    print(f"Found {len(custom_plots)} custom plot(s) from strategy")
            except Exception as e:
                print(f"Warning: Could not create custom plots: {e}")
        
        # Create interactive plots with Bokeh
        try:
            bokeh_creator = BokehInteractivePlotCreator(self.plots_dir)
            
            # Create full interactive analysis
            html_file = bokeh_creator.create_interactive_analysis(
                data, signal_result, results, test_name, show_plot, 
                self.plots_version_manager, custom_plots=custom_plots
            )
            
            if html_file is None:
                print("\033[91mFAILED: Interactive plot creation returned None. Check for data or plotting errors.\033[0m")
                
        except Exception as e:
            print(f"\033[91mFAILED: Error creating interactive plot: {e}\033[0m")
            print("Check your data and plotting configuration.")
    
    def _save_results(self, test_metadata: Dict[str, Any], test_name: str):
        """Save test results to files with versioning"""
        import json
        
        # Ensure directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Get versioned filenames
        versioned_base = self.version_manager.get_versioned_filename(test_name, prefix="V")
        
        # Save performance results as CSV
        results_df = pl.DataFrame([test_metadata['performance_results']])
        results_file = os.path.join(self.results_dir, f"{versioned_base}_results.csv")
        results_df.write_csv(results_file)
        
        # Save metadata as JSON
        metadata_file = os.path.join(self.results_dir, f"{versioned_base}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(test_metadata, f, indent=2, default=str)
        
        # Store versioned filenames for later use
        test_metadata['versioned_files'] = {
            'results_file': results_file,
            'metadata_file': metadata_file,
            'version': versioned_base.split('_')[-1]
        }
    
    def generate_test_report(self, test_metadata: Dict[str, Any], test_name: str = "insample_excellence"):
        """Generate a comprehensive test report"""
        report_content = textwrap.dedent(f"""
            # {test_name.replace('_', ' ').title()} Test Report

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

            - **Total Signals:** {test_metadata['strategy_summary']['total_signals']}
            - **Position Counts:** {test_metadata['strategy_summary']['position_counts']}

            ## Signal Changes

            {test_metadata['strategy_summary']['signal_change_counts']}

            ## Analysis

            *Add your analysis and observations here...*

            ## Next Steps

            *Add next steps and recommendations here...*
        """).strip()
        
        # Use versioned filename for the report
        versioned_base = self.version_manager.get_versioned_filename(test_name, prefix="V")
        report_file = os.path.join(self.results_dir, f"{versioned_base}_report.md")
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"Test report saved to: {report_file}")