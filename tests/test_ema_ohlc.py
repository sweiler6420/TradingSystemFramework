"""EMA / MACD OHLC column validation."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import polars as pl

from framework.features.ema_feature import EmaFeature, validate_ohlc_price_column
from framework.features.macd_feature import MacdFeature


def _sample_ohlcv(n: int = 40) -> pl.DataFrame:
    t0 = datetime(2024, 1, 1, 0, 0, 0)
    return pl.DataFrame(
        {
            "timestamp": [t0 + timedelta(hours=i) for i in range(n)],
            "open": [float(i) for i in range(n)],
            "high": [float(i) + 0.5 for i in range(n)],
            "low": [float(i) - 0.5 for i in range(n)],
            "close": [float(i) + 0.1 for i in range(n)],
            "volume": [1000.0] * n,
        }
    )


class OhlcColumnTests(unittest.TestCase):
    def test_validate_accepts_uppercase(self) -> None:
        self.assertEqual(validate_ohlc_price_column("OPEN"), "open")

    def test_validate_rejects_other_columns(self) -> None:
        with self.assertRaises(ValueError):
            validate_ohlc_price_column("typical_price")

    def test_ema_open_vs_close_differ(self) -> None:
        df = _sample_ohlcv()
        e_close = EmaFeature(df, period=5, column="close").get_values()
        e_open = EmaFeature(df, period=5, column="open").get_values()
        self.assertNotEqual(e_close[-1], e_open[-1])

    def test_macd_passes_column(self) -> None:
        df = _sample_ohlcv()
        m = MacdFeature(df, fast_period=3, slow_period=5, signal_period=2, column="low")
        self.assertEqual(m.column, "low")


if __name__ == "__main__":
    unittest.main()
