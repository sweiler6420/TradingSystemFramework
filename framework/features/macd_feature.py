"""
MACD Feature
============

Moving Average Convergence Divergence: fast EMA minus slow EMA, signal EMA of the MACD line,
and histogram (MACD − signal). Built from :class:`EmaFeature` instances.
"""

from typing import Optional

import numpy as np
import polars as pl

from framework.features.base_feature import BaseFeature
from framework.features.ema_feature import EmaFeature, validate_ohlc_price_column


class MacdFeature(BaseFeature):
    """
    MACD (default 12 / 26 / 9).

    Fast and slow EMAs use one OHLC column (default ``close``); signal EMA is on the
    MACD line. Primary ``calculate()`` result is the MACD line. Use ``get_signal()``
    and ``get_histogram()`` for the other components.
    """

    def __init__(
        self,
        data: pl.DataFrame = None,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = "close",
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.column = validate_ohlc_price_column(column)

        self._macd_line: Optional[pl.Series] = None
        self._signal_line: Optional[pl.Series] = None
        self._histogram: Optional[pl.Series] = None

        super().__init__(
            name="MACD",
            data=data,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            column=self.column,
        )

    def calculate(self) -> pl.Series:
        if self.data is None:
            raise ValueError("No data available for MACD calculation")
        if not self.validate_data(self.data):
            raise ValueError("Data must contain OHLCV columns")
        if self.column not in self.data.columns:
            raise ValueError(f"Column '{self.column}' not in data")

        fast_ema = EmaFeature(
            self.data, period=self.fast_period, column=self.column
        ).get_values()
        slow_ema = EmaFeature(
            self.data, period=self.slow_period, column=self.column
        ).get_values()

        macd_line = fast_ema - slow_ema
        signal_line = EmaFeature(
            self.data,
            period=self.signal_period,
            input_series=macd_line,
        ).get_values()
        histogram = macd_line - signal_line

        self._macd_line = macd_line
        self._signal_line = signal_line
        self._histogram = histogram

        return macd_line

    def get_values(self, recalculate: bool = False) -> pl.Series:
        """MACD line (same as primary feature series)."""
        return super().get_values(recalculate=recalculate)

    def get_macd_line(self) -> pl.Series:
        if self._macd_line is None:
            self.get_values()
        return self._macd_line

    def get_signal(self) -> pl.Series:
        if self._signal_line is None:
            self.get_values()
        return self._signal_line

    def get_histogram(self) -> pl.Series:
        if self._histogram is None:
            self.get_values()
        return self._histogram

    def get_plot(self, x_range=None, **kwargs):
        """
        MACD + signal lines and histogram (Bokeh), aligned with other feature plots via ``x_range``.
        """
        if self.data is None:
            raise ValueError("No data available for MACD plotting")

        try:
            from bokeh.models import NumeralTickFormatter, Range1d
            from bokeh.plotting import figure

            macd = self.get_macd_line()
            signal = self.get_signal()
            hist = self.get_histogram()
            ts = self.data["timestamp"]

            p = figure(
                title=(
                    f"MACD ({self.fast_period}, {self.slow_period}, {self.signal_period}) "
                    f"[{self.column}]"
                ),
                x_axis_type="datetime",
                height=300,
                sizing_mode="stretch_width",
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above",
            )

            if x_range is not None:
                p.x_range = x_range

            # Histogram bar width (ms) for datetime x-axis
            if len(ts) > 1:
                t64 = ts.cast(pl.Datetime).to_numpy()
                diffs = np.diff(t64.astype("datetime64[ns]").astype(np.int64))
                w = float(np.median(np.abs(diffs))) * 0.8 / 1e6
            else:
                w = 3600000.0

            colors = []
            for h in hist.to_list():
                try:
                    v = float(h)
                except (TypeError, ValueError):
                    v = 0.0
                colors.append("#26a69a" if v >= 0 else "#ef5350")
            p.vbar(
                x=ts,
                width=w,
                bottom=0,
                top=hist,
                fill_color=colors,
                line_color=colors,
                alpha=0.85,
                legend_label="Histogram",
            )

            p.line(ts, macd, line_color="#2962ff", line_width=2, legend_label="MACD")
            p.line(ts, signal, line_color="#ff6d00", line_width=2, legend_label="Signal")

            p.line(ts, [0.0] * len(ts), line_color="gray", line_dash="dashed", line_width=1)

            combined = np.concatenate(
                [macd.to_numpy(), signal.to_numpy(), hist.to_numpy()]
            )
            finite = combined[np.isfinite(combined)]
            if finite.size:
                ymin, ymax = float(finite.min()), float(finite.max())
            else:
                ymin, ymax = 0.0, 1.0
            if ymin == ymax:
                ymin, ymax = ymin - 1.0, ymax + 1.0
            pad = (ymax - ymin) * 0.08 + 1e-9
            p.y_range = Range1d(ymin - pad, ymax + pad)

            p.legend.location = "top_left"
            p.legend.click_policy = "hide"

            p.yaxis.formatter = NumeralTickFormatter(format="0,0.0000")

            return p
        except ImportError:
            print("Warning: Bokeh not available for MACD plotting")
            return None
