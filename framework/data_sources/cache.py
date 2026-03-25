"""
Parquet cache paths and ``ensure_cached`` for provider-fetched OHLCV.

Cache files live under a per-project directory (e.g. ``research/my_strategy/data/``).
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Union

from framework.data_handling.market_session import SessionPolicy, session_cache_tag
from framework.data_sources.protocol import MarketDataProvider

DateLike = Union[str, date, datetime]


def _parse_date(d: DateLike) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return date.fromisoformat(d.strip()[:10])
    raise TypeError(f"Unsupported date type: {type(d)!r}")


def safe_symbol_label(symbol: str) -> str:
    """Filesystem-safe fragment derived from a ticker symbol."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", symbol.replace("/", "-"))


def safe_interval_label(interval: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", interval)


def safe_provider_label(name: str) -> str:
    """Filesystem-safe fragment for a provider id (e.g. ``massive``, ``yfinance``)."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip().lower())


def _provider_key_from_instance(provider: MarketDataProvider) -> str:
    cls = type(provider).__name__
    if cls.endswith("Provider"):
        cls = cls[:-8]
    return safe_provider_label(cls.lower())


def cache_parquet_path(
    cache_dir: str | Path,
    *,
    symbol: str,
    interval: str,
    provider: str,
    start: date,
    end: date,
    session_policy: SessionPolicy | None = None,
) -> Path:
    """Deterministic cache filename: symbol, interval, provider, dates, optional session tag."""
    pk = safe_provider_label(provider)
    base = (
        f"{safe_symbol_label(symbol)}_{safe_interval_label(interval)}_{pk}_"
        f"{start.isoformat()}_{end.isoformat()}"
    )
    if session_policy is not None:
        base += f"_{session_cache_tag(session_policy)}"
    name = f"{base}.parquet"
    return Path(cache_dir) / name


def ensure_cached(
    provider: MarketDataProvider,
    *,
    symbol: str,
    interval: str,
    start: DateLike,
    end: DateLike,
    cache_dir: str | Path,
    force_refresh: bool = False,
    session_policy: SessionPolicy | None = None,
    provider_key: str | None = None,
) -> Path:
    """
    Return path to a Parquet file containing OHLCV for ``[start, end)``.

    If the file already exists and ``force_refresh`` is False, the provider is not
    called. Otherwise data is fetched, written, and the path is returned.

    The cache filename includes a **provider** segment (``provider_key``, or derived
    from the provider class name) so different APIs do not share the same Parquet
    file for the same symbol and dates.

    ``session_policy`` is embedded in the filename when set (or taken from
    ``provider.session_policy`` when the provider exposes it), so RTH and extended
    caches do not overwrite each other.
    """
    sd = _parse_date(start)
    ed = _parse_date(end)
    effective_policy = session_policy
    if effective_policy is None:
        effective_policy = getattr(provider, "session_policy", None)
    pk = provider_key if provider_key is not None else _provider_key_from_instance(provider)
    path = cache_parquet_path(
        cache_dir,
        symbol=symbol,
        interval=interval,
        provider=pk,
        start=sd,
        end=ed,
        session_policy=effective_policy,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force_refresh:
        return path
    df = provider.fetch_historical(symbol, interval, sd, ed)
    if df.is_empty():
        raise ValueError(
            f"No rows returned for {symbol!r} interval={interval!r} "
            f"start={sd.isoformat()} end={ed.isoformat()}"
        )
    df.write_parquet(path)
    return path
