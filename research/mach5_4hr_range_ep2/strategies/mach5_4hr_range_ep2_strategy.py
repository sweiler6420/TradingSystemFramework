"""
Mach54HrRangeEp2Strategy - Mach5_4Hr_Range_Ep2 Strategy
=======================================================

Strategy implementation for mach5_4hr_range_ep2 research project.

Rules (New York / America/New_York timezone):
- Range window: 00:00–03:59 ET each day.  Max high and min low of that window
  define the daily Range High and Range Low (drawn as white lines on the chart).
- No trades are taken during the 00:00–03:59 ET window.

Long entry
----------
1. Price CLOSES BELOW the Range Low after the range window.
2. Collect the lowest LOW of every bar that closes outside (below) the range
   as the candidate stop level.
3. On the NEXT candle whose close is back INSIDE the range (close >= Range Low),
   enter LONG at that close.
4. Stop Loss  = lowest low of the outside-range candles.
5. Take Profit = entry + 2 × (entry − stop)  [2:1 RR].

Short entry (mirror of Long)
-----------------------------
1. Price CLOSES ABOVE the Range High.
2. Collect the highest HIGH of bars outside (above) the range.
3. On the NEXT close back INSIDE the range (close <= Range High), enter SHORT.
4. Stop Loss  = highest high of the outside-range candles.
5. Take Profit = entry − 2 × (stop − entry).

Exits: stop or take-profit hit intrabar (stop checked before TP).
Setups reset at the start of each new NY calendar day.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import polars as pl

from framework import SignalBasedStrategy, SignalChange, SignalResult
from framework.signals import calculate_strategy_returns
from framework.risk_reward import (
    TradeRiskConfig,
    long_exit_intrabar,
    long_exit_series,
    long_take_profit_price,
    short_exit_intrabar,
    short_exit_series,
    short_take_profit_price,
)

_RR = TradeRiskConfig(risk_reward_ratio=2.0)

# States for the local replay tracker inside generate_raw_signal
_NEUTRAL = 0
_WATCHING_LONG = 1
_WATCHING_SHORT = 2
_IN_LONG = 3
_IN_SHORT = 4


class Mach54HrRangeEp2Strategy(SignalBasedStrategy):
    """Mach5 4-hour range EP2 strategy."""

    @staticmethod
    def normalize_data(data: pl.DataFrame) -> pl.DataFrame:
        """Convert timestamps from naive UTC to naive America/New_York so the
        chart x-axis and all date/hour logic share the same ET clock."""
        return data.with_columns(
            pl.col("timestamp")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("America/New_York")
            .dt.replace_time_zone(None)
            .alias("timestamp")
        )

    def __init__(self, data: pl.DataFrame):
        super().__init__("Mach5_4Hr_Range_Ep2", data)
        self._range_high: Optional[np.ndarray] = None
        self._range_low: Optional[np.ndarray] = None
        self._plot_stop_series: Optional[pl.Series] = None
        self._plot_tp_series: Optional[pl.Series] = None
        # Per-entry-bar stop prices for generate_exit_conditions replay
        self._entry_stop_long: list[float] = []
        self._entry_stop_short: list[float] = []

    # ------------------------------------------------------------------
    # Range calculation
    # ------------------------------------------------------------------

    def _build_ny_df(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add _date_ny and _hour_ny columns to *data*.

        ``normalize_data`` already converted timestamps to naive ET, so we read
        date and hour directly without any further timezone conversion.
        """
        return data.with_columns([
            pl.col("timestamp").dt.date().alias("_date_ny"),
            pl.col("timestamp").dt.hour().cast(pl.Int32).alias("_hour_ny"),
        ])

    def _compute_4hr_range(self, data: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Per-bar arrays of the NY-day 4-hour range high and low (NaN if no data).

        Timestamps are already in naive ET (applied by ``normalize_data``), so
        ``_hour_ny`` is just ``timestamp.hour`` and ``_date_ny`` is ``timestamp.date``.
        """
        df = self._build_ny_df(data)

        daily_range = (
            df.filter(pl.col("_hour_ny") < 4)
            .group_by("_date_ny")
            .agg([
                pl.col("high").max().alias("_range_high"),
                pl.col("low").min().alias("_range_low"),
            ])
        )

        df = df.join(daily_range, on="_date_ny", how="left")
        return df["_range_high"].to_numpy(), df["_range_low"].to_numpy()

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_raw_signal(self, **kwargs) -> pl.Series:
        data = self.data
        n = len(data)

        range_high, range_low = self._compute_4hr_range(data)
        self._range_high = range_high
        self._range_low = range_low

        df_ny = self._build_ny_df(data)
        hours_ny = df_ny["_hour_ny"].to_numpy()
        dates_ny = df_ny["_date_ny"].to_list()

        high = data["high"].to_numpy()
        low = data["low"].to_numpy()
        close = data["close"].to_numpy()

        out: list[SignalChange] = [SignalChange.NO_CHANGE] * n
        stop_vals = [float("nan")] * n
        tp_vals = [float("nan")] * n

        # Per-entry-bar stop prices (for generate_exit_conditions)
        entry_stop_long = [float("nan")] * n
        entry_stop_short = [float("nan")] * n

        state = _NEUTRAL
        outside_low = float("nan")   # lowest low while watching long
        outside_high = float("nan")  # highest high while watching short
        active_stop = float("nan")
        active_tp = float("nan")
        prev_date = None

        for i in range(n):
            rh = float(range_high[i])
            rl = float(range_low[i])
            h = float(high[i])
            lo = float(low[i])
            c = float(close[i])
            hour = int(hours_ny[i])
            date = dates_ny[i]

            # --- Day boundary: reset any open setup (not open trades) ---
            if date != prev_date and prev_date is not None:
                if state in (_WATCHING_LONG, _WATCHING_SHORT):
                    state = _NEUTRAL
                    outside_low = outside_high = float("nan")
            prev_date = date

            # ----------------------------------------------------------
            # Manage open long position — checked on EVERY bar (including
            # the 00:00–03:59 range-finding window) so SL/TP is never
            # skipped for an overnight trade.  Only ONE trade can be open
            # at a time; the exclusive state machine guarantees this.
            # ----------------------------------------------------------
            if state == _IN_LONG:
                if math.isfinite(active_stop) and math.isfinite(active_tp):
                    if long_exit_intrabar(lo, h, active_stop, active_tp):
                        state = _NEUTRAL
                        active_stop = active_tp = float("nan")
                        # stop_vals[i] stays NaN (exit bar)
                        continue
                    stop_vals[i] = active_stop
                    tp_vals[i] = active_tp
                continue

            # ----------------------------------------------------------
            # Manage open short position — same reasoning as above.
            # ----------------------------------------------------------
            if state == _IN_SHORT:
                if math.isfinite(active_stop) and math.isfinite(active_tp):
                    if short_exit_intrabar(lo, h, active_stop, active_tp):
                        state = _NEUTRAL
                        active_stop = active_tp = float("nan")
                        continue
                    stop_vals[i] = active_stop
                    tp_vals[i] = active_tp
                continue

            # --- Range-finding window: no NEW entries ---
            if hour < 4:
                continue

            # --- Need a valid range for new entries ---
            if not (math.isfinite(rh) and math.isfinite(rl)):
                continue

            # ----------------------------------------------------------
            # Watching for long re-entry
            # ----------------------------------------------------------
            if state == _WATCHING_LONG:
                if c < rl:
                    # Still outside: extend the outside-range low
                    outside_low = min(outside_low, lo)
                else:
                    # First close back inside → LONG entry
                    sl = outside_low
                    if math.isfinite(sl) and sl < c:
                        tp = long_take_profit_price(c, sl, _RR.risk_reward_ratio)
                        out[i] = SignalChange.NEUTRAL_TO_LONG
                        entry_stop_long[i] = sl
                        active_stop = sl
                        active_tp = tp
                        stop_vals[i] = sl
                        tp_vals[i] = tp
                        state = _IN_LONG
                    else:
                        state = _NEUTRAL
                    outside_low = float("nan")
                continue

            # ----------------------------------------------------------
            # Watching for short re-entry
            # ----------------------------------------------------------
            if state == _WATCHING_SHORT:
                if c > rh:
                    # Still outside: extend the outside-range high
                    outside_high = max(outside_high, h)
                else:
                    # First close back inside → SHORT entry
                    sl = outside_high
                    if math.isfinite(sl) and sl > c:
                        tp = short_take_profit_price(c, sl, _RR.risk_reward_ratio)
                        out[i] = SignalChange.NEUTRAL_TO_SHORT
                        entry_stop_short[i] = sl
                        active_stop = sl
                        active_tp = tp
                        stop_vals[i] = sl
                        tp_vals[i] = tp
                        state = _IN_SHORT
                    else:
                        state = _NEUTRAL
                    outside_high = float("nan")
                continue

            # ----------------------------------------------------------
            # NEUTRAL: watch for initial breakout
            # ----------------------------------------------------------
            if c < rl:
                state = _WATCHING_LONG
                outside_low = lo
            elif c > rh:
                state = _WATCHING_SHORT
                outside_high = h

        self._plot_stop_series = pl.Series(stop_vals)
        self._plot_tp_series = pl.Series(tp_vals)
        self._entry_stop_long = entry_stop_long
        self._entry_stop_short = entry_stop_short

        return pl.Series(out)

    def generate_exit_conditions(
        self,
        data: pl.DataFrame,
        raw_signals: Optional[pl.Series] = None,
        **kwargs,
    ) -> Optional[pl.Series]:
        """SL/TP exits replayed from the entry-bar stop levels."""
        if raw_signals is None:
            return None

        close = data["close"].to_numpy()

        long_ex = long_exit_series(
            data,
            raw_signals,
            entry_price_fn=lambda i: float(close[i]),
            stop_price_fn=lambda i: float(self._entry_stop_long[i]),
            risk_config=_RR,
        )
        short_ex = short_exit_series(
            data,
            raw_signals,
            entry_price_fn=lambda i: float(close[i]),
            stop_price_fn=lambda i: float(self._entry_stop_short[i]),
            risk_config=_RR,
        )
        return pl.Series(
            [bool(x) or bool(y) for x, y in zip(long_ex.to_list(), short_ex.to_list())]
        )

    # ------------------------------------------------------------------
    # Returns (use intrabar RR fill prices, not bar close)
    # ------------------------------------------------------------------

    def _calculate_strategy_returns(
        self, data: pl.DataFrame, signal_result: SignalResult
    ) -> pl.Series:
        if (
            self._plot_stop_series is not None
            and self._plot_tp_series is not None
            and len(self._plot_stop_series) == len(data)
        ):
            return calculate_strategy_returns(
                data,
                signal_result,
                stop_loss=self._plot_stop_series,
                take_profit=self._plot_tp_series,
            )
        return super()._calculate_strategy_returns(data, signal_result)

    # ------------------------------------------------------------------
    # Plot hooks
    # ------------------------------------------------------------------

    def get_trade_levels_for_plot(
        self,
        data: pl.DataFrame,
        signal_result: SignalResult,
    ) -> Optional[tuple[pl.Series, pl.Series]]:
        if self._plot_stop_series is None or self._plot_tp_series is None:
            return None
        if len(self._plot_stop_series) != len(data):
            return None
        return (self._plot_stop_series, self._plot_tp_series)

    def add_price_overlays(
        self,
        price_figure,
        data: pl.DataFrame,
        signal_result: SignalResult,
        *,
        results: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Draw the daily 4-hour range high and low as white lines."""
        if self._range_high is None or self._range_low is None:
            return
        if "timestamp" not in data.columns:
            return

        ts = data["timestamp"]

        price_figure.line(
            ts,
            self._range_high,
            line_width=1.5,
            line_color="#ffffff",
            line_alpha=0.85,
            legend_label="4Hr Range High",
        )
        price_figure.line(
            ts,
            self._range_low,
            line_width=1.5,
            line_color="#ffffff",
            line_alpha=0.85,
            legend_label="4Hr Range Low",
        )
