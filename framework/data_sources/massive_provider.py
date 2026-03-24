"""
Historical OHLCV via Massive.com REST API (official ``massive`` Python client).

Polygon/Massive ``limit`` applies to **base aggregates** used to build each result bar
(see ``list_aggs`` docstring: max 50_000 base aggregates). Intraday bars are built from
**minute** bases (e.g. one 1h bar ≈ 60 minute bases); daily+ bars use **day** bases.

We **disable** the client's internal pagination (``pagination=False``) and split the
requested ``[start, end)`` range into **custom windows** so each ``list_aggs`` call stays
under one response page — avoiding chained HTTP requests that exhaust rate limits.

Set ``MASSIVE_API_KEY`` in the environment (do not commit keys to source control).
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, time as dt_time, timedelta, timezone
from typing import Any, Dict, Tuple

import polars as pl

from framework.data_handling.market_session import SessionPolicy
from framework.data_sources.errors import MassiveDataError

# API max ``limit`` on base aggregates queried per request (Massive / Polygon).
MASSIVE_AGG_PAGE_LIMIT = 50_000

# Pause between our own ``list_aggs`` calls (one HTTP request each when pagination is off).
MASSIVE_CHUNK_DELAY_SECONDS = 15.0

# yfinance-style interval strings → (multiplier, Massive timespan)
_INTERVAL_TO_MASSIVE: Dict[str, Tuple[int, str]] = {
    "1m": (1, "minute"),
    "2m": (2, "minute"),
    "5m": (5, "minute"),
    "15m": (15, "minute"),
    "30m": (30, "minute"),
    "60m": (60, "minute"),
    "90m": (90, "minute"),
    "1h": (1, "hour"),
    "1d": (1, "day"),
    "5d": (5, "day"),
    "1wk": (1, "week"),
    "1mo": (1, "month"),
    "3mo": (3, "month"),
}


def _interval_to_multiplier_timespan(interval: str) -> Tuple[int, str]:
    key = interval.strip().lower()
    if key not in _INTERVAL_TO_MASSIVE:
        supported = ", ".join(sorted(_INTERVAL_TO_MASSIVE.keys()))
        raise MassiveDataError(
            f"Unsupported interval {interval!r} for Massive provider. "
            f"Supported: {supported}"
        )
    return _INTERVAL_TO_MASSIVE[key]


def _to_date(d: date | datetime) -> date:
    if isinstance(d, datetime):
        return d.date()
    return d


def _agg_ms_to_naive_utc(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)


def _base_units_per_output_bar(multiplier: int, timespan: str) -> int:
    """
    Approximate base aggregate units **per output bar** for Polygon's ``limit`` budget.

    - Intraday: base = **minutes** (an ``hour`` bar uses ``multiplier * 60`` minute bases).
    - Daily and longer: base = **calendar days** (week ≈ 7 days, month ≈ 30, etc.).
    """
    m = max(1, int(multiplier))
    if timespan == "minute":
        return m
    if timespan == "hour":
        return m * 60
    if timespan == "day":
        return m
    if timespan == "week":
        return m * 7
    if timespan == "month":
        return m * 30
    if timespan == "quarter":
        return m * 91
    if timespan == "year":
        return m * 365
    return 1


def _chunk_span_timedelta(
    multiplier: int,
    timespan: str,
    agg_limit: int,
    safety_factor: float,
) -> timedelta:
    """
    Max calendar span for **one** ``list_aggs`` call so base aggregates stay under ``agg_limit``.

    ``max_output_bars ≈ floor(agg_limit * safety / base_units_per_bar)``; span is that many
    output bars laid end-to-end in wall time.
    """
    m = max(1, int(multiplier))
    base_u = _base_units_per_output_bar(m, timespan)
    base_u = max(1, base_u)
    lim = max(1, int(agg_limit))
    n_max = max(1, int(lim * max(0.1, min(1.0, safety_factor)) // base_u))

    if timespan == "minute":
        return timedelta(minutes=n_max * m)
    if timespan == "hour":
        return timedelta(hours=n_max * m)
    if timespan == "day":
        return timedelta(days=n_max * m)
    if timespan == "week":
        return timedelta(weeks=n_max * m)
    if timespan == "month":
        return timedelta(days=n_max * m * 30)
    if timespan == "quarter":
        return timedelta(days=n_max * m * 91)
    if timespan == "year":
        return timedelta(days=n_max * m * 365)
    return timedelta(hours=1)


def _normalize_aggs(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )
    return pl.DataFrame(rows)


class MassiveProvider:
    """
    Fetch historical aggregates using the official ``massive`` package (Massive.com API).

    Parameters
    ----------
    api_key
        Massive API key. If omitted, reads ``MASSIVE_API_KEY`` from the environment.
    session_policy
        Stored for cache tagging / :class:`~framework.data_handling.DataHandler`.
    pagination
        ``False`` (default): one HTTP response per date window — we split ranges ourselves.
        ``True``: SDK follows ``next_url`` (extra requests; can hit 429s).
    agg_page_limit
        ``limit`` query param (max 50_000 base aggregates per Polygon semantics).
    safety_factor
        Fraction of ``agg_page_limit`` to budget per chunk (default ``0.95``) for API slack.
    chunk_delay_seconds
        Pause between our successive ``list_aggs`` calls (default :data:`MASSIVE_CHUNK_DELAY_SECONDS`).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        session_policy: SessionPolicy = SessionPolicy.CRYPTO_UTC_24H,
        pagination: bool = False,
        agg_page_limit: int = MASSIVE_AGG_PAGE_LIMIT,
        safety_factor: float = 0.95,
        chunk_delay_seconds: float = MASSIVE_CHUNK_DELAY_SECONDS,
    ) -> None:
        self.session_policy = session_policy
        self.pagination = pagination
        self.agg_page_limit = max(1, min(int(agg_page_limit), MASSIVE_AGG_PAGE_LIMIT))
        self.safety_factor = max(0.1, min(1.0, float(safety_factor)))
        self.chunk_delay_seconds = max(0.0, chunk_delay_seconds)
        self._api_key = api_key

    def _client(self):
        try:
            from massive import RESTClient
        except ImportError as e:
            raise MassiveDataError(
                "The 'massive' package is required. Install with: pip install massive"
            ) from e
        key = self._api_key if self._api_key is not None else os.environ.get("MASSIVE_API_KEY")
        if not key or not str(key).strip():
            raise MassiveDataError(
                "Massive API key missing. Set environment variable MASSIVE_API_KEY "
                "or pass api_key=... to MassiveProvider()."
            )
        return RESTClient(api_key=str(key).strip(), pagination=self.pagination)

    def fetch_historical(
        self,
        symbol: str,
        interval: str,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        start_d = _to_date(start)
        end_d = _to_date(end)
        if start_d >= end_d:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                }
            )

        mult, timespan = _interval_to_multiplier_timespan(interval)
        client = self._client()

        chunk_span = _chunk_span_timedelta(
            mult, timespan, self.agg_page_limit, self.safety_factor
        )

        # [start, end) in naive UTC (aligned with Yahoo-style bars in this framework).
        cur = datetime.combine(start_d, dt_time(0, 0, 0))
        end_exclusive = datetime.combine(end_d, dt_time(0, 0, 0))

        rows: list[dict[str, Any]] = []
        chunk_idx = 0
        try:
            while cur < end_exclusive:
                chunk_end = min(cur + chunk_span, end_exclusive)
                for a in client.list_aggs(
                    ticker=symbol,
                    multiplier=mult,
                    timespan=timespan,
                    from_=cur,
                    to=chunk_end,
                    limit=self.agg_page_limit,
                ):
                    rows.append(
                        {
                            "timestamp": _agg_ms_to_naive_utc(int(a.timestamp)),
                            "open": float(a.open),
                            "high": float(a.high),
                            "low": float(a.low),
                            "close": float(a.close),
                            "volume": float(a.volume) if a.volume is not None else 0.0,
                        }
                    )
                chunk_idx += 1
                cur = chunk_end
                if cur < end_exclusive and self.chunk_delay_seconds > 0:
                    time.sleep(self.chunk_delay_seconds)
        except Exception as e:
            raise MassiveDataError(
                f"Massive list_aggs failed for {symbol!r} interval={interval!r} "
                f"start={start_d.isoformat()} end={end_d.isoformat()} "
                f"(after chunk {chunk_idx}): {e}"
            ) from e

        out = _normalize_aggs(rows)
        if out.is_empty():
            return out

        out = out.sort("timestamp").unique(subset=["timestamp"], keep="last")
        out = out.filter(
            (pl.col("timestamp").dt.date() >= start_d)
            & (pl.col("timestamp").dt.date() < end_d)
        )
        return out
