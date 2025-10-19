"""
Mach2_Rsi_Testing Research Script
====================================

Main research script for mach2_rsi_testing.
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
from strategies.mach2_rsi_testing_strategy import Mach2RsiTestingStrategy


def run_insample_excellence_test():
    """Run in-sample excellence test (proof of concept)"""
    print("=== MACH2_RSI_TESTING RESEARCH - IN-SAMPLE EXCELLENCE TEST ===")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    data_handler = DataHandler('framework/data/BTCUSD1hour.pq')
    data_handler.load_data()
    data_handler.filter_date_range(2023, 2024)
    data = data_handler.get_data()
    
    # Create strategy
    strategy = Mach2RsiTestingStrategy()
    
    # Initialize the standardized test with strategy
    test = InSampleExcellenceTest(os.path.dirname(__file__), strategy)
    
    # Run the test
    test_metadata = test.run_test(data_handler, "insample_excellence")
    
    # Create plots
    signal_result = strategy.generate_signals()
    test.create_performance_plots(data, signal_result, test_metadata['performance_results'])
    
    # # Generate report
    test.generate_test_report(test_metadata)
    
    print(f"\n=== MACH2_RSI_TESTING RESEARCH COMPLETED ===")
    print("Check the following directories for results:")
    print("- results/ - Performance metrics and metadata")
    print("- plots/ - Visualization charts")
    print("- README.md - Project documentation")
    
    return False


def main():
    """Main research function"""
    print("Starting mach2_rsi_testing research...")
    
    # Run in-sample excellence test
    results = run_insample_excellence_test()
    
    print(f"\nmach2_rsi_testing research completed!")
    print("Check the results/ and plots/ directories for outputs.")


if __name__ == "__main__":
    main()
