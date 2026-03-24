"""
Historical OHLCV via yfinance with chunked requests, retries, and polite delays.

Intraday history is requested in slices to respect typical yfinance limits; ranges
are merged and de-duplicated by ``timestamp``.
"""

from __future__ import annotations

import time
from datetime import date, datetime as dt, timedelta, timezone
from typing import List

import pandas as pd
import polars as pl
import yfinance as yf

from framework.data_handling.market_session import SessionPolicy
from framework.data_sources.errors import YFinanceDataError
from framework.data_sources.retry import retry_with_backoff


def _yfinance_prepost(session_policy: SessionPolicy) -> bool:
    """Whether to request Yahoo pre/post market bars (extended session)."""
    return session_policy == SessionPolicy.US_EQUITY_EXTENDED

# Yahoo Finance limits intraday (minute/hour) history to roughly the last 730 calendar days.
YAHOO_INTRADAY_LOOKBACK_DAYS = 730

# Intervals treated as daily or longer (not subject to the 730-day intraday rule).
_DAILY_OR_LONGER_INTERVALS = frozenset({"1d", "5d", "1wk", "1mo", "3mo"})


def _to_date(d: date | dt) -> date:
    if isinstance(d, dt):
        return d.date()
    return d


def _is_intraday_interval(interval: str) -> bool:
    return interval.strip().lower() not in _DAILY_OR_LONGER_INTERVALS


def _utc_today() -> date:
    return dt.now(timezone.utc).date()


def _validate_yahoo_intraday_range(symbol: str, interval: str, start_d: date, end_d: date) -> None:
    """
    Yahoo rejects intraday history outside a rolling ~730-day window (see API errors).

    Fail fast with a clear message instead of many empty chunk requests.
    """
    if not _is_intraday_interval(interval):
        return
    today = _utc_today()
    oldest = today - timedelta(days=YAHOO_INTRADAY_LOOKBACK_DAYS - 1)
    if end_d > today:
        raise YFinanceDataError(
            f"yfinance interval {interval!r}: end date {end_d} is after today (UTC) {today}. "
            f"Use end <= today or switch to a daily interval for historical backtests."
        )
    if start_d > today:
        raise YFinanceDataError(
            f"yfinance interval {interval!r}: start date {start_d} is in the future (UTC today={today})."
        )
    if start_d < oldest:
        raise YFinanceDataError(
            f"yfinance interval {interval!r}: intraday data is only available for roughly the last "
            f"{YAHOO_INTRADAY_LOOKBACK_DAYS} days (Yahoo). Requested start={start_d} is before "
            f"{oldest}. Either narrow the range to [ {oldest}, {end_d} ), or use interval '1d' "
            f"for longer history (symbol={symbol!r})."
        )


def _normalize_history_df(pdf: pd.DataFrame) -> pl.DataFrame:
    """Convert yfinance pandas history to Polars with DataHandler-friendly columns."""
    if pdf is None or pdf.empty:
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

    pdf = pdf.copy()
    pdf = pdf.reset_index()
    # Index becomes 'Date' or 'Datetime'
    time_col = None
    for c in pdf.columns:
        cl = str(c).lower()
        if cl in ("date", "datetime", "index"):
            time_col = c
            break
    if time_col is None:
        time_col = pdf.columns[0]

    rename_map = {}
    for c in pdf.columns:
        cl = str(c).lower()
        if cl in ("open", "high", "low", "close", "volume"):
            rename_map[c] = cl
        elif c == time_col:
            rename_map[c] = "timestamp"

    out = pdf.rename(columns=rename_map)
    keep = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in out.columns]
    out = out[keep]
    plf = pl.from_pandas(out)
    if plf.is_empty():
        return plf

    plf = plf.with_columns(pl.col("timestamp").cast(pl.Datetime))
    return plf


class YFinanceProvider:
    """
    Fetch historical bars using yfinance.

    Parameters
    ----------
    chunk_days
        Length of each sub-range when splitting a long request (intraday-friendly).
    max_retries
        Retries per chunk on transient failures.
    base_backoff_seconds
        Base delay for exponential backoff between retries.
    min_interval_seconds
        Minimum sleep after each successful chunk (before the next chunk) to reduce load.
    session_policy
        Drives Yahoo ``prepost`` in ``history()``: ``True`` only for ``US_EQUITY_EXTENDED``
        (pre/post and extended session where Yahoo applies it). Use that policy for forex
        or other symbols when you want the extended fetch; ``CRYPTO_UTC_24H`` and
        ``US_EQUITY_RTH`` use ``prepost=False``. After load, ``apply_session_policy`` still
        keeps all rows for ``US_EQUITY_EXTENDED`` (same as 24h crypto) — only ``US_EQUITY_RTH``
        filters by exchange hours.
    """

    def __init__(
        self,
        *,
        session_policy: SessionPolicy = SessionPolicy.CRYPTO_UTC_24H,
        chunk_days: int = 60,
        max_retries: int = 5,
        base_backoff_seconds: float = 1.0,
        min_interval_seconds: float = 0.75,
    ) -> None:
        self.session_policy = session_policy
        self.chunk_days = max(1, chunk_days)
        self.max_retries = max(1, max_retries)
        self.base_backoff_seconds = base_backoff_seconds
        self.min_interval_seconds = min_interval_seconds

    def _fetch_one_chunk(
        self,
        symbol: str,
        interval: str,
        chunk_start: date,
        chunk_end: date,
    ) -> pl.DataFrame:
        """yfinance ``end`` is exclusive for daily+; use next day for inclusivity of last bar."""

        def _call() -> pd.DataFrame:
            t = yf.Ticker(symbol)
            # yfinance ``end`` is exclusive (bars strictly before ``end`` date).
            return t.history(
                start=chunk_start.isoformat(),
                end=chunk_end.isoformat(),
                interval=interval,
                auto_adjust=False,
                prepost=_yfinance_prepost(self.session_policy),
            )

        try:
            pdf = retry_with_backoff(
                _call,
                max_retries=self.max_retries,
                base_sleep_seconds=self.base_backoff_seconds,
            )
        except Exception as e:
            raise YFinanceDataError(
                f"yfinance failed for {symbol!r} interval={interval!r} "
                f"chunk [{chunk_start} -> {chunk_end}): {e}"
            ) from e
        return _normalize_history_df(pdf)

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

        _validate_yahoo_intraday_range(symbol, interval, start_d, end_d)

        # Daily+ can use larger effective chunks to cut round trips
        eff_chunk = self.chunk_days
        if interval in ("1d", "5d", "1wk", "1mo", "3mo"):
            eff_chunk = max(self.chunk_days, 365 * 5)

        chunks = []
        cur = start_d
        delta = timedelta(days=eff_chunk)
        while cur < end_d:
            nxt = min(cur + delta, end_d)
            chunks.append((cur, nxt))
            cur = nxt

        frames: List[pl.DataFrame] = []
        for i, (cs, ce) in enumerate(chunks):
            part = self._fetch_one_chunk(symbol, interval, cs, ce)
            if not part.is_empty():
                frames.append(part)
            if i < len(chunks) - 1:
                time.sleep(self.min_interval_seconds)

        if not frames:
            raise YFinanceDataError(
                f"No OHLCV returned for {symbol!r} interval={interval!r} "
                f"start={start_d.isoformat()} end={end_d.isoformat()}. "
                f"The symbol may be invalid or delisted, or Yahoo returned no rows for every chunk "
                f"(for intraday, ensure the range is within the last ~730 days; try interval '1d')."
            )

        merged = pl.concat(frames).sort("timestamp")
        merged = merged.unique(subset=["timestamp"], keep="last")

        # Trim to requested window [start_d, end_d) on calendar dates
        merged = merged.filter(
            (pl.col("timestamp").dt.date() >= start_d)
            & (pl.col("timestamp").dt.date() < end_d)
        )
        if merged.is_empty():
            raise YFinanceDataError(
                f"No rows left after filtering for {symbol!r} interval={interval!r} "
                f"start={start_d.isoformat()} end={end_d.isoformat()}."
            )
        return merged
