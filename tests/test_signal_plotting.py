"""Tests for SignalResult plotting helpers (RR exit prices vs close)."""

from __future__ import annotations

import unittest

import polars as pl

from framework.signals import SignalChange, SignalResult


class SignalPlottingTests(unittest.TestCase):
    def test_exit_marker_uses_rr_fill_not_close_when_levels_provided(self) -> None:
        data = pl.DataFrame(
            {
                "timestamp": [0, 1, 2],
                "open": [100.0, 100.0, 100.0],
                "high": [101.0, 101.0, 100.0],
                "low": [99.0, 99.0, 89.0],
                "close": [100.0, 100.0, 95.0],
            }
        )
        signal_changes = pl.Series(
            [
                SignalChange.NO_CHANGE,
                SignalChange.NEUTRAL_TO_LONG,
                SignalChange.LONG_TO_NEUTRAL,
            ]
        )
        position_signals = pl.Series([0, 1, 0])
        sr = SignalResult(
            position_signals=position_signals,
            signal_changes=signal_changes,
        )
        stop_loss = pl.Series([float("nan"), 90.0, float("nan")])
        take_profit = pl.Series([float("nan"), 120.0, float("nan")])
        plot_df = sr.get_signal_changes_for_plotting(
            data, stop_loss=stop_loss, take_profit=take_profit
        )
        exit_prices = [
            float(r["price"])
            for r in plot_df.iter_rows(named=True)
            if r["signal_change"] == SignalChange.LONG_TO_NEUTRAL
        ]
        self.assertEqual(len(exit_prices), 1)
        # Wick to 89 vs stop 90 → fill at stop, not close 95
        self.assertAlmostEqual(exit_prices[0], 90.0)

    def test_exit_marker_falls_back_to_close_without_levels(self) -> None:
        data = pl.DataFrame(
            {
                "timestamp": [0, 1, 2],
                "close": [100.0, 100.0, 95.0],
            }
        )
        signal_changes = pl.Series(
            [
                SignalChange.NO_CHANGE,
                SignalChange.NEUTRAL_TO_LONG,
                SignalChange.LONG_TO_NEUTRAL,
            ]
        )
        sr = SignalResult(
            position_signals=pl.Series([0, 1, 0]),
            signal_changes=signal_changes,
        )
        plot_df = sr.get_signal_changes_for_plotting(data)
        exit_prices = [
            float(r["price"])
            for r in plot_df.iter_rows(named=True)
            if r["signal_change"] == SignalChange.LONG_TO_NEUTRAL
        ]
        self.assertAlmostEqual(exit_prices[0], 95.0)


if __name__ == "__main__":
    unittest.main()
