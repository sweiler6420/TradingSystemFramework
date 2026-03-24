"""Tests for framework.data_sources cache + provider wiring."""

from __future__ import annotations

import unittest
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from framework.data_handling.market_session import SessionPolicy
from framework.data_sources.cache import cache_parquet_path, ensure_cached


def _sample_ohlcv() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2023, 6, 15, 12, 0, 0)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1_000.0],
        }
    )


class FakeProvider:
    def __init__(self) -> None:
        self.calls = 0

    def fetch_historical(self, symbol: str, interval: str, start: date, end: date) -> pl.DataFrame:
        self.calls += 1
        assert symbol == "TEST"
        assert interval == "1h"
        return _sample_ohlcv()


class CacheParquetPathTests(unittest.TestCase):
    def test_cache_parquet_path_deterministic(self) -> None:
        p = cache_parquet_path(
            "cache_dir",
            symbol="BTC-USD",
            interval="1h",
            start=date(2023, 1, 1),
            end=date(2024, 1, 1),
        )
        self.assertEqual(p.name, "BTC-USD_1h_2023-01-01_2024-01-01.parquet")
        self.assertEqual(p.parent, Path("cache_dir"))

    def test_cache_parquet_path_includes_session_tag(self) -> None:
        p = cache_parquet_path(
            "cache_dir",
            symbol="AAPL",
            interval="1h",
            start=date(2024, 6, 1),
            end=date(2026, 1, 1),
            session_policy=SessionPolicy.US_EQUITY_RTH,
        )
        self.assertEqual(
            p.name,
            "AAPL_1h_2024-06-01_2026-01-01_us_equity_rth.parquet",
        )


class EnsureCachedTests(unittest.TestCase):
    def test_second_call_skips_fetch(self) -> None:
        with TemporaryDirectory() as td:
            prov = FakeProvider()
            p = ensure_cached(
                prov,
                symbol="TEST",
                interval="1h",
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                cache_dir=td,
            )
            self.assertTrue(Path(p).is_file())
            self.assertEqual(prov.calls, 1)
            df = pl.read_parquet(p)
            self.assertEqual(set(df.columns), {"timestamp", "open", "high", "low", "close", "volume"})

            ensure_cached(
                prov,
                symbol="TEST",
                interval="1h",
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                cache_dir=td,
            )
            self.assertEqual(prov.calls, 1)

    def test_force_refresh_refetches(self) -> None:
        with TemporaryDirectory() as td:
            prov = FakeProvider()
            ensure_cached(
                prov,
                symbol="TEST",
                interval="1h",
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                cache_dir=td,
            )
            ensure_cached(
                prov,
                symbol="TEST",
                interval="1h",
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                cache_dir=td,
                force_refresh=True,
            )
            self.assertEqual(prov.calls, 2)


if __name__ == "__main__":
    unittest.main()
