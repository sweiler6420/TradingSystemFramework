"""
Pluggable market data providers and Parquet caching.

Example::

    from datetime import date
    from framework.data_sources import YFinanceProvider, ensure_cached
    from framework import DataHandler

    project_root = ...
    cache_dir = f"{project_root}/data"
    path = ensure_cached(
        YFinanceProvider(),
        symbol="AAPL",
        interval="1h",
        start=date(2023, 1, 1),
        end=date(2024, 1, 1),
        cache_dir=cache_dir,
    )
    dh = DataHandler(str(path))
    dh.load_data()
"""

from framework.data_handling.market_session import SessionPolicy
from framework.data_sources.cache import cache_parquet_path, ensure_cached
from framework.data_sources.errors import YFinanceDataError
from framework.data_sources.protocol import MarketDataProvider
from framework.data_sources.retry import retry_with_backoff
from framework.data_sources.yfinance_provider import YFinanceProvider

__all__ = [
    "MarketDataProvider",
    "SessionPolicy",
    "YFinanceDataError",
    "YFinanceProvider",
    "ensure_cached",
    "cache_parquet_path",
    "retry_with_backoff",
]
