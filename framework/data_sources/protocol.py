"""
Market data provider protocol.

Implementations (e.g. yfinance, Polygon) return Polars frames with:
``timestamp``, ``open``, ``high``, ``low``, ``close``, ``volume`` (lowercase).
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class MarketDataProvider(Protocol):
    def fetch_historical(
        self,
        symbol: str,
        interval: str,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """Fetch OHLCV bars for ``[start, end)`` (end exclusive) or provider convention."""
        ...
