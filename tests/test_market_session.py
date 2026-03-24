"""Tests for framework.data_handling.market_session."""

from __future__ import annotations

import unittest
from datetime import datetime

import polars as pl

from framework.data_handling.market_session import SessionPolicy, apply_session_policy


class ApplySessionPolicyTests(unittest.TestCase):
    def test_us_equity_rth_keeps_weekday_session_bar(self) -> None:
        # Naive UTC: 13:30 UTC on Mon Jun 3 2024 = 9:30 AM EDT (RTH open bar).
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 6, 3, 13, 30),
                    datetime(2024, 6, 3, 12, 0),
                    datetime(2024, 6, 1, 16, 0),
                ],
                "x": [1, 2, 3],
            }
        )
        out = apply_session_policy(
            df, SessionPolicy.US_EQUITY_RTH, naive_timestamp_tz="UTC"
        )
        self.assertEqual(out.height, 1)
        self.assertEqual(out["x"].item(), 1)

    def test_crypto_unchanged_row_count(self) -> None:
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 6, 1, 12, 0)],
                "x": [1],
            }
        )
        out = apply_session_policy(df, SessionPolicy.CRYPTO_UTC_24H)
        self.assertEqual(out.height, 1)


if __name__ == "__main__":
    unittest.main()
