"""Tests for Massive.com market data provider (no live API calls)."""

from __future__ import annotations

import os
import unittest
from datetime import date

from framework.data_sources.errors import MassiveDataError
from datetime import timedelta

from framework.data_sources.massive_provider import (
    MASSIVE_AGG_PAGE_LIMIT,
    MassiveProvider,
    _base_units_per_output_bar,
    _chunk_span_timedelta,
    _interval_to_multiplier_timespan,
)


class MassiveProviderTests(unittest.TestCase):
    def test_interval_maps_1h_to_hour(self) -> None:
        self.assertEqual(_interval_to_multiplier_timespan("1h"), (1, "hour"))

    def test_chunk_span_one_hour_respects_base_aggregate_budget(self) -> None:
        """1h bars cost ~60 minute bases each; one chunk must not exceed the budget."""
        span = _chunk_span_timedelta(1, "hour", MASSIVE_AGG_PAGE_LIMIT, 0.95)
        n_hours = int(span.total_seconds() // 3600)
        base_per_bar = _base_units_per_output_bar(1, "hour")
        self.assertEqual(base_per_bar, 60)
        self.assertLessEqual(
            n_hours * base_per_bar,
            int(MASSIVE_AGG_PAGE_LIMIT * 0.95),
        )
        self.assertEqual(span, timedelta(hours=791))

    def test_unsupported_interval_raises(self) -> None:
        with self.assertRaises(MassiveDataError):
            _interval_to_multiplier_timespan("2h")

    def test_empty_range_returns_empty_frame(self) -> None:
        p = MassiveProvider(api_key="dummy-not-used")
        df = p.fetch_historical("C:EURUSD", "1h", date(2024, 6, 1), date(2024, 6, 1))
        self.assertTrue(df.is_empty())

    def test_missing_api_key_raises(self) -> None:
        old = os.environ.pop("MASSIVE_API_KEY", None)
        try:
            p = MassiveProvider(api_key=None)
            with self.assertRaises(MassiveDataError):
                p.fetch_historical(
                    "C:EURUSD", "1h", date(2024, 6, 1), date(2024, 6, 2)
                )
        finally:
            if old is not None:
                os.environ["MASSIVE_API_KEY"] = old


if __name__ == "__main__":
    unittest.main()
