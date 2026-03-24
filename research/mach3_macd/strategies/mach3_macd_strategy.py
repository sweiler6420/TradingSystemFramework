"""
Mach3MacdStrategy - MACD crossover (long-only)
==============================================

Enters long when MACD crosses above the signal line; exits when MACD crosses below.
"""

import polars as pl

from framework import SignalBasedStrategy, SignalChange, MacdFeature


class Mach3MacdStrategy(SignalBasedStrategy):
    """MACD / signal crossover — buy on bullish cross, flat on bearish cross."""

    def __init__(
        self,
        data: pl.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = "close",
    ):
        super().__init__("Mach3 MACD Crossover", data)
        self.macd_feature = MacdFeature(
            data,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            column=column,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.column = self.macd_feature.column
        self.position = 0

    def generate_raw_signal(self, **kwargs) -> pl.Series:
        """Bullish cross → enter long; bearish cross → exit to neutral."""
        macd_line = self.macd_feature.get_macd_line()
        signal_line = self.macd_feature.get_signal()

        n = len(self.data)
        signals_list = [SignalChange.NO_CHANGE] * n

        for i in range(1, n):
            m_curr = macd_line[i]
            m_prev = macd_line[i - 1]
            s_curr = signal_line[i]
            s_prev = signal_line[i - 1]

            if any(
                x is None or (isinstance(x, float) and str(x) == "nan")
                for x in (m_curr, m_prev, s_curr, s_prev)
            ):
                continue

            m_curr = float(m_curr)
            m_prev = float(m_prev)
            s_curr = float(s_curr)
            s_prev = float(s_prev)

            # Bullish crossover: MACD crosses above signal
            if m_prev <= s_prev and m_curr > s_curr and self.position == 0:
                signals_list[i] = SignalChange.NEUTRAL_TO_LONG
                self.position = 1

            # Bearish crossover: MACD crosses below signal
            elif m_prev >= s_prev and m_curr < s_curr and self.position == 1:
                signals_list[i] = SignalChange.LONG_TO_NEUTRAL
                self.position = 0

        return pl.Series(signals_list)

    def create_custom_plots(self, data: pl.DataFrame, signal_result, **kwargs) -> list:
        plots = []
        macd_plot = self.macd_feature.get_plot()
        if macd_plot is not None:
            plots.append(macd_plot)
        return plots
