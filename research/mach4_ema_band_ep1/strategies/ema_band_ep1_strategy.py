"""
EmaBandEp1Strategy — EMA channel trend-reversal (long & short)
=============================================================

Uses two :class:`framework.sequential_fsm.SequentialSetup` machines with **mirrored**
rules: long setup after sustained trade below EMA200 then cross up; short setup after
sustained trade above EMA200 then cross down. Bar-count timeouts apply per side.
See ``framework/sequential_fsm.py`` for FSM semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import polars as pl

from framework import SignalBasedStrategy, SignalChange, SignalResult
from framework.signals import calculate_strategy_returns
from framework.features.ema_feature import EmaFeature
from framework.risk_reward import (
    TradeRiskConfig,
    long_exit_intrabar,
    long_exit_series,
    long_take_profit_price,
    short_exit_intrabar,
    short_exit_series,
    short_take_profit_price,
)
from framework.sequential_fsm import SequentialSetup, StepResult

# Bar periods (same units as your bar timeframe).
EMA_FAST = 5
EMA_SLOW = 100
EMA_TREND = 200

# Staged-setup windows (bars = ticks on whatever timeframe you load, e.g. 1h → hours).
MAX_BARS_WAIT_TREND_CHANGE = 25
MAX_BARS_WAIT_PULLBACK = 50
MAX_BARS_WAIT_TREND_BREAKOUT = 25

# Prior bars required on the correct side of EMA200 before the first cross into the setup.
PRIOR_BARS_CLOSE_BELOW_EMA200 = 25
PRIOR_BARS_CLOSE_ABOVE_EMA200 = 25

COLOR_EMA_FAST = "#e6edf3"  # white channel
COLOR_EMA_SLOW = "#3fb950"  # green channel
COLOR_EMA_TREND = "#f0883e"  # orange trend

META_LONG_ENTRY = "long_entry"
META_SHORT_ENTRY = "short_entry"

DEFAULT_TRADE_RISK = TradeRiskConfig(risk_reward_ratio=2.0)


class EP1SetupState(Enum):
    """Staged entry FSM (shared shape for long and short machines)."""

    IDLE = auto()
    WAIT_TREND_CHANGE = auto()
    WAIT_PULLBACK = auto()
    WAIT_TREND_BREAKOUT = auto()


@dataclass(frozen=True)
class EP1Context:
    """Per-bar inputs for FSM predicates (OHLC + EMAs + prior-bar cross + prior-run filters)."""

    open: float
    high: float
    low: float
    close: float
    prev_close: float
    prev_ema200: float
    ema5_high: float
    ema5_low: float
    ema100_high: float
    ema100_low: float
    ema200: float
    prior_bars_below_ema200_ok: bool
    prior_bars_above_ema200_ok: bool


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _green_channel_bounds(ema_lo: float, ema_hi: float) -> tuple[float, float]:
    return (min(ema_lo, ema_hi), max(ema_lo, ema_hi))


def _prior_n_closes_below_ema200(close, ema200, i: int, n_prior: int) -> bool:
    if i < n_prior:
        return False
    for k in range(1, n_prior + 1):
        idx = i - k
        if not _finite(float(close[idx])) or not _finite(float(ema200[idx])):
            return False
        if float(close[idx]) >= float(ema200[idx]):
            return False
    return True


def _prior_n_closes_above_ema200(close, ema200, i: int, n_prior: int) -> bool:
    if i < n_prior:
        return False
    for k in range(1, n_prior + 1):
        idx = i - k
        if not _finite(float(close[idx])) or not _finite(float(ema200[idx])):
            return False
        if float(close[idx]) <= float(ema200[idx]):
            return False
    return True


def _bar_breaks_green_channel(
    open_: float,
    high: float,
    low: float,
    close: float,
    ema100_low: float,
    ema100_high: float,
) -> bool:
    g_lo, g_hi = _green_channel_bounds(ema100_low, ema100_high)
    bar_lo = min(open_, high, low, close)
    bar_hi = max(open_, high, low, close)
    return bar_hi >= g_lo and bar_lo <= g_hi


def _build_long_setup_fsm() -> SequentialSetup[EP1SetupState]:
    S = EP1SetupState
    fsm: SequentialSetup[EP1SetupState] = SequentialSetup(S.IDLE)

    fsm.add_transition(
        S.IDLE,
        lambda c: (
            c.prev_close < c.prev_ema200
            and c.close > c.ema200
            and c.prior_bars_below_ema200_ok
        ),
        S.WAIT_TREND_CHANGE,
    )
    fsm.add_transition(
        S.WAIT_TREND_CHANGE,
        lambda c: c.ema5_high > c.ema200 and c.ema5_low > c.ema200,
        S.WAIT_PULLBACK,
    )
    fsm.add_transition(
        S.WAIT_PULLBACK,
        lambda c: _bar_breaks_green_channel(
            c.open, c.high, c.low, c.close, c.ema100_low, c.ema100_high
        ),
        S.WAIT_TREND_BREAKOUT,
    )
    fsm.add_transition(
        S.WAIT_TREND_BREAKOUT,
        lambda c: c.close > max(c.ema5_high, c.ema5_low),
        S.IDLE,
        meta=META_LONG_ENTRY,
    )
    return fsm


def _build_short_setup_fsm() -> SequentialSetup[EP1SetupState]:
    """Mirror of long: cross down from above, white below orange, green touch, close below white."""
    S = EP1SetupState
    fsm: SequentialSetup[EP1SetupState] = SequentialSetup(S.IDLE)

    fsm.add_transition(
        S.IDLE,
        lambda c: (
            c.prev_close > c.prev_ema200
            and c.close < c.ema200
            and c.prior_bars_above_ema200_ok
        ),
        S.WAIT_TREND_CHANGE,
    )
    fsm.add_transition(
        S.WAIT_TREND_CHANGE,
        lambda c: c.ema5_high < c.ema200 and c.ema5_low < c.ema200,
        S.WAIT_PULLBACK,
    )
    fsm.add_transition(
        S.WAIT_PULLBACK,
        lambda c: _bar_breaks_green_channel(
            c.open, c.high, c.low, c.close, c.ema100_low, c.ema100_high
        ),
        S.WAIT_TREND_BREAKOUT,
    )
    fsm.add_transition(
        S.WAIT_TREND_BREAKOUT,
        lambda c: c.close < min(c.ema5_high, c.ema5_low),
        S.IDLE,
        meta=META_SHORT_ENTRY,
    )
    return fsm


def _update_bars_in_state(
    fsm: SequentialSetup[EP1SetupState], res: StepResult[EP1SetupState], bars: int
) -> int:
    if res.invalidated:
        return 0
    if res.transitioned:
        return 0 if fsm.state == EP1SetupState.IDLE else 1
    if fsm.state != EP1SetupState.IDLE:
        return bars + 1
    return 0


def _apply_bar_count_timeouts(
    fsm: SequentialSetup[EP1SetupState], bars: int
) -> tuple[int, bool]:
    st = fsm.state
    if st == EP1SetupState.WAIT_TREND_CHANGE and bars >= MAX_BARS_WAIT_TREND_CHANGE:
        fsm.reset()
        return 0, True
    if st == EP1SetupState.WAIT_PULLBACK and bars >= MAX_BARS_WAIT_PULLBACK:
        fsm.reset()
        return 0, True
    if st == EP1SetupState.WAIT_TREND_BREAKOUT and bars >= MAX_BARS_WAIT_TREND_BREAKOUT:
        fsm.reset()
        return 0, True
    return bars, False


def _freeze_opposite_fsm(
    long_fsm: SequentialSetup[EP1SetupState],
    short_fsm: SequentialSetup[EP1SetupState],
) -> None:
    """Only one staged setup at a time: if one side is active, hold the other in IDLE."""
    if long_fsm.state != EP1SetupState.IDLE and short_fsm.state != EP1SetupState.IDLE:
        short_fsm.reset()
    elif long_fsm.state != EP1SetupState.IDLE:
        short_fsm.reset()
    elif short_fsm.state != EP1SetupState.IDLE:
        long_fsm.reset()


class EmaBandEp1Strategy(SignalBasedStrategy):
    """EMA band long/short; at most one open position; staged entry FSMs idle while exposed."""

    def __init__(self, data: pl.DataFrame, *, trade_risk: TradeRiskConfig = DEFAULT_TRADE_RISK):
        super().__init__("EMA Band EP1", data)

        self.ema5_high = EmaFeature(data, period=EMA_FAST, column="high").get_values()
        self.ema5_low = EmaFeature(data, period=EMA_FAST, column="low").get_values()
        self.ema100_high = EmaFeature(data, period=EMA_SLOW, column="high").get_values()
        self.ema100_low = EmaFeature(data, period=EMA_SLOW, column="low").get_values()
        self.ema200_close = EmaFeature(data, period=EMA_TREND, column="close").get_values()

        self._trade_risk = trade_risk
        self._long_fsm = _build_long_setup_fsm()
        self._short_fsm = _build_short_setup_fsm()
        self._active_rr_stop: float | None = None
        self._active_rr_tp: float | None = None
        self._plot_stop_series: pl.Series | None = None
        self._plot_tp_series: pl.Series | None = None

    def generate_raw_signal(self, **kwargs) -> pl.Series:
        n = len(self.data)
        out: list[SignalChange] = [SignalChange.NO_CHANGE] * n
        position = 0
        long_bars = 0
        short_bars = 0
        self._active_rr_stop = self._active_rr_tp = None

        stop_vals = [float("nan")] * n
        tp_vals = [float("nan")] * n

        open_ = self.data["open"].to_numpy()
        close = self.data["close"].to_numpy()
        low = self.data["low"].to_numpy()
        high = self.data["high"].to_numpy()
        e5h = self.ema5_high.to_numpy()
        e5l = self.ema5_low.to_numpy()
        e1h = self.ema100_high.to_numpy()
        e1l = self.ema100_low.to_numpy()
        e200 = self.ema200_close.to_numpy()

        def _snap_trade_levels(i: int) -> None:
            if (
                position in (1, -1)
                and self._active_rr_stop is not None
                and self._active_rr_tp is not None
            ):
                stop_vals[i] = float(self._active_rr_stop)
                tp_vals[i] = float(self._active_rr_tp)
            else:
                stop_vals[i] = float("nan")
                tp_vals[i] = float("nan")

        for i in range(1, n):
            c = float(close[i])
            o = float(open_[i])
            lo = float(low[i])
            hi = float(high[i])
            pc = float(close[i - 1])
            p200 = float(e200[i - 1])
            if not all(
                _finite(x)
                for x in (
                    e5h[i],
                    e5l[i],
                    e1h[i],
                    e1l[i],
                    e200[i],
                    c,
                    lo,
                    hi,
                    o,
                    pc,
                    p200,
                )
            ):
                continue

            prior_below = _prior_n_closes_below_ema200(
                close, e200, i, PRIOR_BARS_CLOSE_BELOW_EMA200
            )
            prior_above = _prior_n_closes_above_ema200(
                close, e200, i, PRIOR_BARS_CLOSE_ABOVE_EMA200
            )
            ctx = EP1Context(
                open=o,
                high=hi,
                low=lo,
                close=c,
                prev_close=pc,
                prev_ema200=p200,
                ema5_high=float(e5h[i]),
                ema5_low=float(e5l[i]),
                ema100_high=float(e1h[i]),
                ema100_low=float(e1l[i]),
                ema200=float(e200[i]),
                prior_bars_below_ema200_ok=prior_below,
                prior_bars_above_ema200_ok=prior_above,
            )

            if position == 1:
                self._long_fsm.reset()
                self._short_fsm.reset()
                long_bars = 0
                short_bars = 0
                if self._active_rr_stop is not None and self._active_rr_tp is not None:
                    if long_exit_intrabar(
                        lo, hi, self._active_rr_stop, self._active_rr_tp
                    ):
                        position = 0
                        self._active_rr_stop = self._active_rr_tp = None
                        self._long_fsm.reset()
                        self._short_fsm.reset()
                _snap_trade_levels(i)
                continue

            if position == -1:
                self._long_fsm.reset()
                self._short_fsm.reset()
                long_bars = 0
                short_bars = 0
                if self._active_rr_stop is not None and self._active_rr_tp is not None:
                    if short_exit_intrabar(
                        lo, hi, self._active_rr_stop, self._active_rr_tp
                    ):
                        position = 0
                        self._active_rr_stop = self._active_rr_tp = None
                        self._long_fsm.reset()
                        self._short_fsm.reset()
                _snap_trade_levels(i)
                continue

            _freeze_opposite_fsm(self._long_fsm, self._short_fsm)

            res_long = self._long_fsm.step(ctx)
            # Do not advance short setup while long is staged or long entry completed this bar.
            skip_short = self._long_fsm.state != EP1SetupState.IDLE or (
                res_long.transitioned and res_long.meta == META_LONG_ENTRY
            )
            if skip_short:
                self._short_fsm.reset()
                res_short = StepResult(self._short_fsm.state, invalidated=False, transitioned=False)
            else:
                res_short = self._short_fsm.step(ctx)
                if self._short_fsm.state != EP1SetupState.IDLE or (
                    res_short.transitioned and res_short.meta == META_SHORT_ENTRY
                ):
                    self._long_fsm.reset()

            long_bars = _update_bars_in_state(self._long_fsm, res_long, long_bars)
            long_bars, _ = _apply_bar_count_timeouts(self._long_fsm, long_bars)

            short_bars = _update_bars_in_state(self._short_fsm, res_short, short_bars)
            short_bars, _ = _apply_bar_count_timeouts(self._short_fsm, short_bars)

            if res_long.meta == META_LONG_ENTRY and res_long.transitioned:
                ep = c
                sp = float(e1l[i])
                if sp < ep:
                    out[i] = SignalChange.NEUTRAL_TO_LONG
                    position = 1
                    self._active_rr_stop = sp
                    self._active_rr_tp = long_take_profit_price(
                        ep, sp, self._trade_risk.risk_reward_ratio
                    )
                _snap_trade_levels(i)
                continue

            if res_short.meta == META_SHORT_ENTRY and res_short.transitioned:
                ep = c
                sp = float(e1h[i])
                if sp > ep:
                    out[i] = SignalChange.NEUTRAL_TO_SHORT
                    position = -1
                    self._active_rr_stop = sp
                    self._active_rr_tp = short_take_profit_price(
                        ep, sp, self._trade_risk.risk_reward_ratio
                    )
                _snap_trade_levels(i)
                continue

            _snap_trade_levels(i)

        self._plot_stop_series = pl.Series(stop_vals)
        self._plot_tp_series = pl.Series(tp_vals)
        return pl.Series(out)

    def get_trade_levels_for_plot(
        self,
        data: pl.DataFrame,
        signal_result: SignalResult,
    ) -> tuple[pl.Series, pl.Series] | None:
        """Per-bar RR stop / take profit (NaN when flat) for Bokeh price hover."""
        if self._plot_stop_series is None or self._plot_tp_series is None:
            return None
        if len(self._plot_stop_series) != len(data):
            return None
        return (self._plot_stop_series, self._plot_tp_series)

    def _calculate_strategy_returns(
        self, data: pl.DataFrame, signal_result: SignalResult
    ) -> pl.Series:
        """Use stop/limit fill prices on exit bars (not bar close) when RR series exist."""
        if self._plot_stop_series is not None and self._plot_tp_series is not None:
            if len(self._plot_stop_series) == len(data) == len(self._plot_tp_series):
                return calculate_strategy_returns(
                    data,
                    signal_result,
                    stop_loss=self._plot_stop_series,
                    take_profit=self._plot_tp_series,
                )
        return super()._calculate_strategy_returns(data, signal_result)

    def add_price_overlays(
        self,
        price_figure,
        data: pl.DataFrame,
        signal_result,
        *,
        results: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Draw EMA5/EMA100 bands and EMA200 on the Bokeh price pane (see ``SignalBasedStrategy``)."""
        if "timestamp" not in data.columns:
            return
        ts = data["timestamp"]
        price_figure.line(
            ts,
            self.ema5_high,
            line_width=1.0,
            line_color=COLOR_EMA_FAST,
            line_alpha=0.75,
            legend_label="EMA5 high",
        )
        price_figure.line(
            ts,
            self.ema5_low,
            line_width=1.0,
            line_color=COLOR_EMA_FAST,
            line_alpha=0.75,
            legend_label="EMA5 low",
        )
        price_figure.line(
            ts,
            self.ema100_high,
            line_width=1.0,
            line_color=COLOR_EMA_SLOW,
            line_alpha=0.75,
            legend_label="EMA100 high",
        )
        price_figure.line(
            ts,
            self.ema100_low,
            line_width=1.0,
            line_color=COLOR_EMA_SLOW,
            line_alpha=0.75,
            legend_label="EMA100 low",
        )
        price_figure.line(
            ts,
            self.ema200_close,
            line_width=1.4,
            line_color=COLOR_EMA_TREND,
            line_alpha=0.9,
            legend_label="EMA200",
        )

    def generate_exit_conditions(
        self,
        data: pl.DataFrame,
        raw_signals: pl.Series | None = None,
        **kwargs,
    ) -> pl.Series | None:
        """Structural exits: long stop at green low / short stop at green high; same RR ratio."""
        if raw_signals is None:
            return None
        close = data["close"].to_numpy()
        ema100_lo = self.ema100_low.to_numpy()
        ema100_hi = self.ema100_high.to_numpy()

        long_ex = long_exit_series(
            data,
            raw_signals,
            entry_price_fn=lambda i: float(close[i]),
            stop_price_fn=lambda i: float(ema100_lo[i]),
            risk_config=self._trade_risk,
        )
        short_ex = short_exit_series(
            data,
            raw_signals,
            entry_price_fn=lambda i: float(close[i]),
            stop_price_fn=lambda i: float(ema100_hi[i]),
            risk_config=self._trade_risk,
        )
        return pl.Series(
            [bool(x) or bool(y) for x, y in zip(long_ex.to_list(), short_ex.to_list())]
        )
