"""
Per-trade stop loss and risk:reward (RR) for long and short strategies
=====================================================================

Strategies set a **stop price** when a trade is opened (often from indicator levels)
and a **strategy-level** :class:`TradeRiskConfig` with ``risk_reward_ratio``.
Take profit is derived so that profit distance = ratio × risk distance.

Examples (1:2 RR): long — entry 100, stop 90 → risk 10 → TP 120.
Short — entry 100, stop 110 → risk 10 → TP 80.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
import polars as pl


# OHLC vs level comparisons (float noise, broker rounding); ~0.1 pip on a ~1.0 FX quote
PRICE_TOUCH_ATOL = 1e-5


def _le_touch(a: float, b: float) -> bool:
    """True if ``a`` is at or below ``b`` within tolerance (for low vs stop, etc.)."""
    return float(a) <= float(b) + PRICE_TOUCH_ATOL


def _ge_touch(a: float, b: float) -> bool:
    """True if ``a`` is at or above ``b`` within tolerance (for high vs TP, etc.)."""
    return float(a) >= float(b) - PRICE_TOUCH_ATOL


@dataclass(frozen=True)
class TradeRiskConfig:
    """
    Reward:risk multiple for long and short trades.

    ``risk_reward_ratio == 2.0`` means you aim to make $2 for each $1 of risk
    (same units as price), i.e. "1:2" in common trader wording.
    """

    risk_reward_ratio: float


def long_risk_per_share(entry_price: float, stop_price: float) -> float:
    """Price distance risked on a long (entry − stop). Must be positive."""
    r = float(entry_price) - float(stop_price)
    if r <= 0:
        raise ValueError(
            "long_risk_per_share requires entry_price > stop_price for a long stop below entry"
        )
    return r


def long_take_profit_price(
    entry_price: float,
    stop_price: float,
    risk_reward_ratio: float,
) -> float:
    """
    Take-profit level for a long: entry + ratio × (entry − stop).

    Matches the example: entry 100, stop 90, ratio 2 → 120.
    """
    risk = long_risk_per_share(entry_price, stop_price)
    return float(entry_price) + float(risk_reward_ratio) * risk


def long_trade_bar_rr_exit_fill(
    bar_low: float,
    bar_high: float,
    stop_price: float,
    take_profit: float,
) -> Optional[float]:
    """
    Per bar while in a long: test stop first (conservative if both touch), else TP.

    Returns the fill price if this bar closes the trade via RR, else ``None``.
    Missing / non-finite take profit is ignored (stop-only path).
    """
    if not math.isfinite(stop_price):
        return None
    if _le_touch(bar_low, stop_price):
        return float(stop_price)
    if math.isfinite(take_profit) and _ge_touch(bar_high, take_profit):
        return float(take_profit)
    return None


def long_exit_intrabar(
    bar_low: float,
    bar_high: float,
    stop_price: float,
    take_profit: float,
) -> bool:
    """True if this bar exits a long via RR (same rules as :func:`long_trade_bar_rr_exit_fill`)."""
    return long_trade_bar_rr_exit_fill(
        bar_low, bar_high, stop_price, take_profit
    ) is not None


def long_exit_fill_price(
    bar_low: float,
    bar_high: float,
    stop_price: float,
    take_profit: float,
) -> float:
    """Fill price for a long RR exit on this bar (stop before TP if both touch)."""
    f = long_trade_bar_rr_exit_fill(bar_low, bar_high, stop_price, take_profit)
    if f is None:
        raise ValueError("long_exit_fill_price: bar does not hit stop or take profit")
    return f


def short_trade_bar_rr_exit_fill(
    bar_low: float,
    bar_high: float,
    stop_price: float,
    take_profit: float,
) -> Optional[float]:
    """
    Per bar while in a short: test stop first, else TP.

    Returns the fill price if this bar closes the trade via RR, else ``None``.
    """
    if not math.isfinite(stop_price):
        return None
    if _ge_touch(bar_high, stop_price):
        return float(stop_price)
    if math.isfinite(take_profit) and _le_touch(bar_low, take_profit):
        return float(take_profit)
    return None


def short_exit_fill_price(
    bar_low: float,
    bar_high: float,
    stop_price: float,
    take_profit: float,
) -> float:
    """Fill price for a short RR exit on this bar (stop before TP if both touch)."""
    f = short_trade_bar_rr_exit_fill(bar_low, bar_high, stop_price, take_profit)
    if f is None:
        raise ValueError("short_exit_fill_price: bar does not hit stop or take profit")
    return f


def adjust_forward_returns_for_rr_exit_fills(
    forward_ret: np.ndarray,
    pos: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    close: np.ndarray,
    stop_s: np.ndarray,
    tp_s: np.ndarray,
) -> None:
    """
    In-place: replace forward log returns on the **last held bar** before each RR exit
    so the terminal move uses stop/limit fill prices instead of the bar close.

    ``forward_ret[i]`` is interpreted as ``log(close[i+1]) - log(close[i])`` (one-bar
    forward return). When the first flat bar after a trade is ``exit_bar``, we set
    ``forward_ret[exit_bar - 1] = log(exit_fill) - log(close[exit_bar - 1])`` using
    the same stop-then-TP rule as :func:`long_trade_bar_rr_exit_fill` /
    :func:`short_trade_bar_rr_exit_fill` on the exit bar's range.
    """
    n = len(pos)
    if not (
        len(forward_ret) == n
        and len(low) == n
        and len(high) == n
        and len(close) == n
        and len(stop_s) == n
        and len(tp_s) == n
    ):
        return

    for exit_bar in range(1, n):
        if pos[exit_bar] != 0 or pos[exit_bar - 1] == 0:
            continue
        side = int(round(float(pos[exit_bar - 1])))
        if side not in (1, -1):
            continue
        last_held = exit_bar - 1
        sl = float(stop_s[last_held])
        tp = float(tp_s[last_held])
        if not math.isfinite(sl):
            continue

        lo = float(low[exit_bar])
        hi = float(high[exit_bar])
        idx = exit_bar - 1
        pc = float(close[idx])
        if side == 1:
            fill = long_trade_bar_rr_exit_fill(lo, hi, sl, tp)
        else:
            fill = short_trade_bar_rr_exit_fill(lo, hi, sl, tp)
        if fill is None:
            continue
        if not math.isfinite(fill) or fill <= 0 or pc <= 0:
            continue
        forward_ret[idx] = math.log(fill) - math.log(pc)


def _is_neutral_to_long(raw: Any) -> bool:
    if raw is None:
        return False
    v = getattr(raw, "value", raw)
    return str(v) == "NEUTRAL_TO_LONG"


def long_exit_flags_replay(
    *,
    high: Sequence[float],
    low: Sequence[float],
    raw_signals: Sequence[Any],
    entry_price_fn: Callable[[int], float],
    stop_price_fn: Callable[[int], float],
    risk_config: TradeRiskConfig,
    extra_exit: Optional[Callable[[int], bool]] = None,
) -> list[bool]:
    """
    Build per-bar exit flags for a **long-only** path that mirrors :class:`SignalManager`
    ordering: if exit fires while long, that bar does **not** also open a new trade from raw.

    - On ``NEUTRAL_TO_LONG`` at bar *i*, entry price and stop are taken from the callbacks
      (typically using bar *i* OHLC / indicators).
    - While long, exit when ``extra_exit(i)`` (e.g. trend rule) **or** stop/TP on the bar's range.

    If ``stop_price_fn(i) >= entry_price_fn(i)``, RR levels are skipped for that trade
    (invalid long stop); only ``extra_exit`` can close the position.
    """
    n = len(raw_signals)
    if not (len(high) == len(low) == n):
        raise ValueError("high, low, raw_signals must have equal length")

    exit_flags = [False] * n
    position_long = False
    active_stop: Optional[float] = None
    active_tp: Optional[float] = None

    for i in range(n):
        if position_long:
            ex = extra_exit(i) if extra_exit is not None else False
            if ex:
                exit_flags[i] = True
                position_long = False
                active_stop = active_tp = None
                continue

            if active_stop is not None and active_tp is not None:
                if long_exit_intrabar(
                    float(low[i]), float(high[i]), active_stop, active_tp
                ):
                    exit_flags[i] = True
                    position_long = False
                    active_stop = active_tp = None
                    continue

        if not position_long and _is_neutral_to_long(raw_signals[i]):
            ep = float(entry_price_fn(i))
            sp = float(stop_price_fn(i))
            position_long = True
            if sp < ep:
                active_stop = sp
                active_tp = long_take_profit_price(ep, sp, risk_config.risk_reward_ratio)
            else:
                active_stop = active_tp = None

    return exit_flags


def long_exit_series(
    data: pl.DataFrame,
    raw_signals: pl.Series,
    *,
    entry_price_fn: Callable[[int], float],
    stop_price_fn: Callable[[int], float],
    risk_config: TradeRiskConfig,
    extra_exit: Optional[Callable[[int], bool]] = None,
) -> pl.Series:
    """Same as :func:`long_exit_flags_replay`, returns a boolean :class:`polars.Series`."""
    h = data["high"].to_list()
    l = data["low"].to_list()
    r = raw_signals.to_list()
    flags = long_exit_flags_replay(
        high=h,
        low=l,
        raw_signals=r,
        entry_price_fn=entry_price_fn,
        stop_price_fn=stop_price_fn,
        risk_config=risk_config,
        extra_exit=extra_exit,
    )
    return pl.Series(flags)


# --- Short (mirror of long): stop above entry, take profit below -----------------


def short_risk_per_share(entry_price: float, stop_price: float) -> float:
    """Price distance risked on a short (stop − entry). Must be positive."""
    r = float(stop_price) - float(entry_price)
    if r <= 0:
        raise ValueError(
            "short_risk_per_share requires stop_price > entry_price for a short stop above entry"
        )
    return r


def short_take_profit_price(
    entry_price: float,
    stop_price: float,
    risk_reward_ratio: float,
) -> float:
    """Take-profit for a short: entry − ratio × (stop − entry)."""
    risk = short_risk_per_share(entry_price, stop_price)
    return float(entry_price) - float(risk_reward_ratio) * risk


def short_exit_intrabar(
    bar_low: float,
    bar_high: float,
    stop_price: float,
    take_profit: float,
) -> bool:
    """True if this bar exits a short via RR (same rules as :func:`short_trade_bar_rr_exit_fill`)."""
    return short_trade_bar_rr_exit_fill(
        bar_low, bar_high, stop_price, take_profit
    ) is not None


def _is_neutral_to_short(raw: Any) -> bool:
    if raw is None:
        return False
    v = getattr(raw, "value", raw)
    return str(v) == "NEUTRAL_TO_SHORT"


def short_exit_flags_replay(
    *,
    high: Sequence[float],
    low: Sequence[float],
    raw_signals: Sequence[Any],
    entry_price_fn: Callable[[int], float],
    stop_price_fn: Callable[[int], float],
    risk_config: TradeRiskConfig,
    extra_exit: Optional[Callable[[int], bool]] = None,
) -> list[bool]:
    """
    Per-bar exit flags for a **short-only** path (mirrors :func:`long_exit_flags_replay`).
    """
    n = len(raw_signals)
    if not (len(high) == len(low) == n):
        raise ValueError("high, low, raw_signals must have equal length")

    exit_flags = [False] * n
    position_short = False
    active_stop: Optional[float] = None
    active_tp: Optional[float] = None

    for i in range(n):
        if position_short:
            ex = extra_exit(i) if extra_exit is not None else False
            if ex:
                exit_flags[i] = True
                position_short = False
                active_stop = active_tp = None
                continue

            if active_stop is not None and active_tp is not None:
                if short_exit_intrabar(
                    float(low[i]), float(high[i]), active_stop, active_tp
                ):
                    exit_flags[i] = True
                    position_short = False
                    active_stop = active_tp = None
                    continue

        if not position_short and _is_neutral_to_short(raw_signals[i]):
            ep = float(entry_price_fn(i))
            sp = float(stop_price_fn(i))
            position_short = True
            if sp > ep:
                active_stop = sp
                active_tp = short_take_profit_price(ep, sp, risk_config.risk_reward_ratio)
            else:
                active_stop = active_tp = None

    return exit_flags


def short_exit_series(
    data: pl.DataFrame,
    raw_signals: pl.Series,
    *,
    entry_price_fn: Callable[[int], float],
    stop_price_fn: Callable[[int], float],
    risk_config: TradeRiskConfig,
    extra_exit: Optional[Callable[[int], bool]] = None,
) -> pl.Series:
    """Same as :func:`short_exit_flags_replay`, as a boolean :class:`polars.Series`."""
    h = data["high"].to_list()
    l = data["low"].to_list()
    r = raw_signals.to_list()
    flags = short_exit_flags_replay(
        high=h,
        low=l,
        raw_signals=r,
        entry_price_fn=entry_price_fn,
        stop_price_fn=stop_price_fn,
        risk_config=risk_config,
        extra_exit=extra_exit,
    )
    return pl.Series(flags)
