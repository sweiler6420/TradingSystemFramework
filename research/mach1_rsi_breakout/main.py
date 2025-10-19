"""
Mach1_Rsi_Breakout Research Script
====================================

Main research script for mach1_rsi_breakout.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Framework imports
from framework import (
    DataHandler, SignalBasedStrategy,
    RSIFeature, DonchianFeature, PositionState, SignalChange
)
from framework.performance import (
    ProfitFactorMeasure, SharpeRatioMeasure, SortinoRatioMeasure,
    MaxDrawdownMeasure, TotalReturnMeasure, WinRateMeasure
)
from framework.significance_testing import MonteCarloSignificanceTest

# Import the standardized test
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tests.insample_excellence_test import InSampleExcellenceTest

# Import project-specific strategy
from strategies.mach1_rsi_breakout_strategy import Mach1RsiBreakoutStrategy


def run_insample_excellence_test():
    """Run in-sample excellence test (proof of concept)"""
    print("=== MACH1_RSI_BREAKOUT RESEARCH - IN-SAMPLE EXCELLENCE TEST ===")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    data_handler = DataHandler(os.path.join(os.path.dirname(__file__), '..', '..', 'framework', 'data', 'BTCUSD1hour.pq'))
    data_handler.load_data()
    data_handler.filter_date_range(2023, 2024)
    data = data_handler.get_data()
    
    print(f"Data loaded: {data.shape[0]} rows from {data['timestamp'][0]} to {data['timestamp'][-1]}")
    
    # Create strategy with data
    strategy = Mach1RsiBreakoutStrategy(data)
    
    # Initialize the standardized test with strategy
    test = InSampleExcellenceTest(os.path.dirname(__file__), strategy)
    
    # Run the test
    test_metadata = test.run_test(data_handler, "insample_excellence")
    
    # Create plots
    signal_result = strategy.generate_signals()
    test.create_performance_plots(data, signal_result, test_metadata['performance_results'])
    
    # Generate report
    test.generate_test_report(test_metadata)
    
    print(f"\n=== MACH1_RSI_BREAKOUT RESEARCH COMPLETED ===")
    print("Check the following directories for results:")
    print("- results/ - Performance metrics and metadata")
    print("- plots/ - Visualization charts")
    print("- README.md - Project documentation")
    
    return test_metadata


def main():
    """Main research function"""
    print("Starting mach1_rsi_breakout research...")
    
    # Run in-sample excellence test
    results = run_insample_excellence_test()
    
    print(f"\nmach1_rsi_breakout research completed!")
    print("Check the results/ and plots/ directories for outputs.")


if __name__ == "__main__":
    main()
