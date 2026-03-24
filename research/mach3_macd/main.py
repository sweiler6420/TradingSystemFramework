"""
Mach3_Macd Research Script
==========================

Main research script for mach3_macd.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import date, datetime

# Framework imports
from framework import DataHandler
from framework.data_sources import SessionPolicy, YFinanceProvider, ensure_cached

# Import the in-sample excellence suite
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from suites.insample_excellence import InSampleExcellenceSuite

# Import project-specific strategy
from strategies.mach3_macd_strategy import Mach3MacdStrategy


def run_insample_excellence_suite():
    """Run in-sample excellence suite (proof of concept)"""
    print("=== MACH3_MACD RESEARCH - IN-SAMPLE EXCELLENCE SUITE ===")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    project_root = os.path.dirname(__file__)
    cache_dir = os.path.join(project_root, "data")
    # Crypto: 24/7 UTC bars; no US equity RTH strip (see SessionPolicy.CRYPTO_UTC_24H).
    session = SessionPolicy.CRYPTO_UTC_24H
    data_path = ensure_cached(
        YFinanceProvider(session_policy=session),
        symbol="BTC-USD",
        interval="1h",
        start=date(2024, 6, 1),
        end=date(2026, 1, 1),
        cache_dir=cache_dir,
    )
    data_handler = DataHandler(str(data_path), session_policy=session)
    data_handler.load_data()
    data_handler.filter_date_range(2024, 2026)
    data = data_handler.get_data()

    print(
        f"Data loaded: {data.height} rows from {data['timestamp'][0]} to {data['timestamp'][-1]}"
    )

    strategy = Mach3MacdStrategy(data)

    test = InSampleExcellenceSuite(os.path.dirname(__file__), strategy)

    test_metadata = test.run_test(data_handler, "insample_excellence")

    signal_result = strategy.generate_signals()
    test.create_performance_plots(
        data,
        signal_result,
        test_metadata["performance_results"],
        test_metadata=test_metadata,
    )

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

    run_insample_excellence_suite()

    print("\nmach3_macd research completed!")
    print("Check the results/ and plots/ directories for outputs.")


if __name__ == "__main__":
    main()
