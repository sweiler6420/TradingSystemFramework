"""
EMA Feature
===========

Exponential moving average (EMA) on a price column or on an arbitrary aligned series.
Used directly and as the building block for MACD and other stacked indicators.
"""

from typing import Optional

import polars as pl

from framework.features.base_feature import BaseFeature

# Price inputs for row-wise indicators (matches typical OHLC bar columns).
OHLC_PRICE_COLUMNS: frozenset[str] = frozenset({"open", "high", "low", "close"})


def validate_ohlc_price_column(column: str) -> str:
    """
    Return normalized column name (lowercase). Raise if not one of open/high/low/close.

    Used when ``EmaFeature`` smooths a price series from ``data`` (not ``input_series``).
    """
    c = column.strip().lower()
    if c not in OHLC_PRICE_COLUMNS:
        raise ValueError(
            f"column must be one of {sorted(OHLC_PRICE_COLUMNS)}, got {column!r}"
        )
    return c


class EmaFeature(BaseFeature):
    """
    Exponential moving average.

    Uses Polars ``ewm_mean(span=period, adjust=False)``, consistent with common
    charting platforms (span relates to smoothing length; alpha = 2 / (span + 1)).

    When ``input_series`` is not set, ``column`` must be one of ``open``, ``high``,
    ``low``, or ``close`` (default ``close``).
    """

    def __init__(
        self,
        data: pl.DataFrame = None,
        period: int = 12,
        column: str = "close",
        input_series: Optional[pl.Series] = None,
    ):
        """
        Args:
            data: OHLCV DataFrame (required unless only used after set_data).
            period: EMA span (number of periods).
            column: Which OHLC price to smooth when ``input_series`` is ``None``.
                One of ``open``, ``high``, ``low``, ``close``; default ``close``.
            input_series: If set, EMA is applied to this series (row-aligned with ``data``),
                e.g. MACD line for the signal EMA.
        """
        self.period = period
        if input_series is None:
            self.column = validate_ohlc_price_column(column)
        else:
            self.column = column.strip().lower() if isinstance(column, str) else column
        self.input_series = input_series

        super().__init__(
            name="EMA",
            data=data,
            period=period,
            column=self.column,
            uses_input_series=input_series is not None,
        )

    def calculate(self) -> pl.Series:
        if self.data is None and self.input_series is None:
            raise ValueError("No data available for EMA calculation")

        if self.input_series is not None:
            if self.data is not None and len(self.input_series) != len(self.data):
                raise ValueError("input_series length must match data row count")
            return self.input_series.ewm_mean(span=self.period, adjust=False)

        if not self.validate_data(self.data):
            raise ValueError("Data must contain OHLCV columns")
        if self.column not in self.data.columns:
            raise ValueError(f"Column '{self.column}' not in data")

        return self.data.select(
            pl.col(self.column).ewm_mean(span=self.period, adjust=False)
        ).to_series()

    def get_plot(self, x_range=None, **kwargs):
        """
        Plot the EMA line over time (same pattern as RSI for strategy dashboards).
        """
        if self.data is None:
            raise ValueError("No data available for EMA plotting")

        try:
            from bokeh.models import NumeralTickFormatter, Range1d
            from bokeh.plotting import figure

            ema_values = self.get_values()
            title_suffix = f" — {self.column}" if not self.params.get("uses_input_series") else ""

            p = figure(
                title=f"EMA ({self.period}){title_suffix}",
                x_axis_type="datetime",
                height=300,
                sizing_mode="stretch_width",
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above",
            )

            if x_range is not None:
                p.x_range = x_range

            p.line(
                self.data["timestamp"],
                ema_values,
                line_color="navy",
                line_width=2,
                legend_label=f"EMA {self.period}",
            )
            p.legend.location = "top_left"
            p.legend.click_policy = "hide"

            p.yaxis.formatter = NumeralTickFormatter(format="0,0.00")

            # Reasonable default y range from data
            ev = ema_values.drop_nulls()
            if len(ev):
                ymin, ymax = float(ev.min()), float(ev.max())
            else:
                ymin, ymax = 0.0, 1.0
            pad = (ymax - ymin) * 0.05 + 1e-9
            p.y_range = Range1d(ymin - pad, ymax + pad)

            return p
        except ImportError:
            print("Warning: Bokeh not available for EMA plotting")
            return None
