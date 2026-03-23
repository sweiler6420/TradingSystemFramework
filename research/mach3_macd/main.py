"""
Mach3_Macd Research Script
==========================

Main research script for mach3_macd.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import datetime

# Framework imports
from framework import DataHandler

# Import the standardized test
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tests.insample_excellence_test import InSampleExcellenceTest

# Import project-specific strategy
from strategies.mach3_macd_strategy import Mach3MacdStrategy


def run_insample_excellence_test():
    """Run in-sample excellence test (proof of concept)"""
    print("=== MACH3_MACD RESEARCH - IN-SAMPLE EXCELLENCE TEST ===")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    root = os.path.join(os.path.dirname(__file__), "..", "..")
    data_path = os.path.join(root, "framework", "data", "BTCUSD1hour.pq")
    data_handler = DataHandler(data_path)
    data_handler.load_data()
    data_handler.filter_date_range(2023, 2024)
    data = data_handler.get_data()

    print(
        f"Data loaded: {data.height} rows from {data['timestamp'][0]} to {data['timestamp'][-1]}"
    )

    strategy = Mach3MacdStrategy(data)

    test = InSampleExcellenceTest(os.path.dirname(__file__), strategy)

    test_metadata = test.run_test(data_handler, "insample_excellence")

    signal_result = strategy.generate_signals()
    test.create_performance_plots(data, signal_result, test_metadata["performance_results"])

    test.generate_test_report(test_metadata)

    print("\n=== MACH3_MACD RESEARCH COMPLETED ===")
    print("Check the following directories for results:")
    print("- results/ - Performance metrics and metadata")
    print("- plots/ - Visualization charts")
    print("- README.md - Project documentation")

    return test_metadata


def main():
    """Main research function"""
    print("Starting mach3_macd research...")

    run_insample_excellence_test()

    print("\nmach3_macd research completed!")
    print("Check the results/ and plots/ directories for outputs.")


if __name__ == "__main__":
    main()
