"""
Market session and venue semantics for OHLCV bars.

Why this exists
---------------
Crypto trades ~24/7 on UTC; US equities trade on an exchange session (regular
hours vs extended). Backtests and live trading should use the **same** bar
definition. This module makes that choice explicit:

* **What you fetch** (e.g. Yahoo ``prepost=False`` vs ``True``) controls whether
  extended-hours bars exist in the file.
* **What you keep** (``SessionPolicy``) controls which timestamps participate in
  the strategy after load.

Regular hours (RTH) filter
--------------------------
For ``US_EQUITY_RTH`` we keep bars whose **timestamp** falls on a NYSE weekday
(Mon–Fri) and between **09:30 and 16:00** America/New_York (bar **open** time),
inclusive of the usual 1h grid (e.g. 9:30, 10:30, …, 15:30 ET).

**Limitations (typical for lightweight frameworks):**

* **Exchange holidays** and **early closes** are not modeled; a holiday may
  still pass the weekday filter if bad data appears, and early-close days may
  keep bars after the real close. For production, plug in an exchange calendar
  (e.g. ``exchange_calendars``) and filter on session open/close per date.

* **Naive timestamps** use ``naive_timestamp_tz`` as the clock for the raw
  column: **UTC** is typical for Yahoo intraday (then RTH math converts to
  ``America/New_York``). Use ``America/New_York`` only if your file stores
  naive wall times in that zone.

Use the same ``SessionPolicy`` and ``naive_timestamp_tz`` in research and in
any live path that builds bars the same way.
"""

from __future__ import annotations

from enum import Enum
import polars as pl

__all__ = ["SessionPolicy", "apply_session_policy", "session_cache_tag"]


class SessionPolicy(str, Enum):
    """Which bars belong to the strategy universe after load."""

    CRYPTO_UTC_24H = "crypto_utc_24h"
    """All loaded bars; timestamps treated as continuous (typically UTC)."""

    US_EQUITY_RTH = "us_equity_rth"
    """US listed equity, regular session only (strip extended hours if present)."""

    US_EQUITY_EXTENDED = "us_equity_extended"
    """Pre-market + RTH + after-hours (as returned by the provider)."""


def session_cache_tag(policy: SessionPolicy) -> str:
    """Short, filesystem-safe fragment for cache filenames."""
    return policy.value


def _expr_in_ny_session_clock(ts: pl.Expr, naive_timestamp_tz: str) -> pl.Expr:
    """
    Express ``ts`` in ``America/New_York`` for RTH filtering.

    Yahoo intraday OHLCV is usually **naive UTC**; use ``naive_timestamp_tz="UTC"``.
    """
    ts = ts.cast(pl.Datetime)
    if naive_timestamp_tz.upper() == "UTC":
        return ts.dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York")
    return ts.dt.replace_time_zone(naive_timestamp_tz)


def _us_rth_mask(ts_local: pl.Expr) -> pl.Expr:
    """Mon–Fri, 09:30 <= time < 16:00 in *local* session clock."""
    wd = ts_local.dt.weekday()
    # Polars: Monday=1 .. Sunday=7
    is_weekday = (wd >= 1) & (wd <= 5)
    # Cast hour/minute off i8 before arithmetic (9*60=540 overflows i8 in Polars).
    h = ts_local.dt.hour().cast(pl.Int32)
    mi = ts_local.dt.minute().cast(pl.Int32)
    sec = ts_local.dt.second().cast(pl.Float64)
    minutes = h * 60 + mi + sec / 60.0
    open_min = 9 * 60 + 30
    close_min = 16 * 60  # 16:00 ET (exclusive upper bound for bar starts)
    in_rth = (minutes >= open_min) & (minutes < close_min)
    return is_weekday & in_rth


def apply_session_policy(
    df: pl.DataFrame,
    policy: SessionPolicy,
    *,
    timestamp_col: str = "timestamp",
    naive_timestamp_tz: str = "UTC",
) -> pl.DataFrame:
    """
    Return a copy of ``df`` with rows restricted to ``policy``.

    For ``US_EQUITY_RTH``, naive timestamps are interpreted in ``naive_timestamp_tz``
    before applying the RTH mask. For ``CRYPTO_UTC_24H`` and ``US_EQUITY_EXTENDED``,
    returns ``df`` unchanged (aside from sorting on ``timestamp_col`` if present).
    """
    if df.is_empty() or timestamp_col not in df.columns:
        return df

    if policy in (SessionPolicy.CRYPTO_UTC_24H, SessionPolicy.US_EQUITY_EXTENDED):
        return df.sort(timestamp_col)

    if policy == SessionPolicy.US_EQUITY_RTH:
        ts = pl.col(timestamp_col)
        ts_local = _expr_in_ny_session_clock(ts, naive_timestamp_tz)
        return (
            df.sort(timestamp_col)
            .filter(_us_rth_mask(ts_local))
            .sort(timestamp_col)
        )

    raise ValueError(f"Unknown SessionPolicy: {policy!r}")
