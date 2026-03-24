"""
Mach4 EMA Band EP1 — research script
====================================

Fetches OHLCV for EUR/USD forex (hourly aggregates via Massive.com / Polygon API).
Set ``MASSIVE_API_KEY`` in the environment (see Massive / Polygon account).
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import date, datetime

# ``research/tests`` (InSampleExcellenceTest, Bokeh report)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from framework import DataHandler
from framework.data_sources import MassiveProvider, SessionPolicy, ensure_cached

from tests.insample_excellence_test import InSampleExcellenceTest

from strategies.ema_band_ep1_strategy import EmaBandEp1Strategy

# Massive forex ticker (C: prefix). Multiplier 1 × hour maps from interval ``1h``.
EURUSD_SYMBOL = "C:EURUSD"
BAR_INTERVAL = "1h"


def run_insample_excellence_test():
    """Run in-sample excellence test (proof of concept)."""
    print("=== MACH4 EMA BAND EP1 — IN-SAMPLE EXCELLENCE TEST ===")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbol: {EURUSD_SYMBOL}  interval: {BAR_INTERVAL}")

    project_root = os.path.dirname(__file__)
    cache_dir = os.path.join(project_root, "data")
    # 24h UTC: keep all FX bars (Massive aggregates are UTC; no US RTH filter).
    session = SessionPolicy.CRYPTO_UTC_24H
    print(f"Session policy: {session.value} (Massive FX / all bars)")
    data_path = ensure_cached(
        MassiveProvider(session_policy=session),
        symbol=EURUSD_SYMBOL,
        interval=BAR_INTERVAL,
        start=date(2024, 3, 1),
        end=date(2026, 3, 1),
        cache_dir=cache_dir,
    )
    data_handler = DataHandler(str(data_path), session_policy=session)
    data_handler.load_data()
    data_handler.filter_date_range(2024, 2027)
    data = data_handler.get_data()

    print(
        f"Data loaded: {data.height} rows from {data['timestamp'][0]} to {data['timestamp'][-1]}"
    )

    strategy = EmaBandEp1Strategy(data)

    test = InSampleExcellenceTest(os.path.dirname(__file__), strategy)

    test_metadata = test.run_test(data_handler, "insample_excellence")

    signal_result = strategy.generate_signals()
    test.create_performance_plots(data, signal_result, test_metadata["performance_results"])

    test.generate_test_report(test_metadata)

    print("\n=== MACH4 EMA BAND EP1 RESEARCH COMPLETED ===")
    print("Check the following directories for results:")
    print("- results/ - Performance metrics and metadata")
    print("- plots/ - Visualization charts")
    print("- README.md - Project documentation")

    return test_metadata


def main():
    print("Starting mach4_ema_band_ep1 research...")
    run_insample_excellence_test()
    print("\nmach4_ema_band_ep1 research completed!")
    print("Check the results/ and plots/ directories for outputs.")


if __name__ == "__main__":
    main()
