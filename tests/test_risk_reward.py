"""Tests for framework.risk_reward."""

from __future__ import annotations

import math
import unittest

import numpy as np
import polars as pl

from framework.risk_reward import (
    TradeRiskConfig,
    adjust_forward_returns_for_rr_exit_fills,
    long_exit_fill_price,
    long_exit_flags_replay,
    long_exit_intrabar,
    long_take_profit_price,
    short_exit_fill_price,
    short_take_profit_price,
)
from framework.signals import SignalChange


class RiskRewardTests(unittest.TestCase):
    def test_long_take_profit_example(self) -> None:
        # Entry 100, stop 90, 1:2 → TP 120
        self.assertAlmostEqual(long_take_profit_price(100.0, 90.0, 2.0), 120.0)

    def test_short_take_profit_example(self) -> None:
        # Short entry 100, stop 110, 1:2 → risk 10 → TP 80
        self.assertAlmostEqual(short_take_profit_price(100.0, 110.0, 2.0), 80.0)

    def test_long_exit_intrabar_stop_only(self) -> None:
        self.assertTrue(long_exit_intrabar(89.0, 100.0, 90.0, 120.0))

    def test_long_exit_intrabar_tp_only(self) -> None:
        self.assertTrue(long_exit_intrabar(95.0, 121.0, 90.0, 120.0))

    def test_long_exit_intrabar_both_prefers_stop(self) -> None:
        self.assertTrue(long_exit_intrabar(89.0, 121.0, 90.0, 120.0))

    def test_long_exit_fill_price_tp_only(self) -> None:
        self.assertAlmostEqual(long_exit_fill_price(95.0, 121.0, 90.0, 120.0), 120.0)

    def test_long_exit_fill_price_stop_only(self) -> None:
        self.assertAlmostEqual(long_exit_fill_price(89.0, 100.0, 90.0, 120.0), 90.0)

    def test_long_exit_fill_price_both_prefers_stop(self) -> None:
        self.assertAlmostEqual(long_exit_fill_price(89.0, 121.0, 90.0, 120.0), 90.0)

    def test_short_exit_fill_price_tp_only(self) -> None:
        self.assertAlmostEqual(short_exit_fill_price(79.0, 100.0, 110.0, 80.0), 80.0)

    def test_short_exit_fill_price_stop_only(self) -> None:
        self.assertAlmostEqual(short_exit_fill_price(80.0, 111.0, 110.0, 80.0), 110.0)

    def test_replay_exit_on_take_profit_bar(self) -> None:
        df = pl.DataFrame(
            {
                "high": [100.0, 100.0, 125.0],
                "low": [99.0, 99.0, 100.0],
                "close": [100.0, 100.0, 125.0],
            }
        )
        raw = pl.Series(
            [
                SignalChange.NO_CHANGE,
                SignalChange.NEUTRAL_TO_LONG,
                SignalChange.NO_CHANGE,
            ]
        )
        cfg = TradeRiskConfig(risk_reward_ratio=2.0)
        flags = long_exit_flags_replay(
            high=df["high"].to_list(),
            low=df["low"].to_list(),
            raw_signals=raw.to_list(),
            entry_price_fn=lambda i: float(df["close"][i]),
            stop_price_fn=lambda i: 90.0,
            risk_config=cfg,
            extra_exit=lambda i: False,
        )
        self.assertFalse(flags[0])
        self.assertFalse(flags[1])
        # tp = 100 + 2*10 = 120; bar index 2 high 125 → exit
        self.assertTrue(flags[2])

    def test_adjust_forward_returns_applies_stop_when_tp_series_is_nan(self) -> None:
        """Previously required finite TP; NaN TP skipped all RR exit fill adjustment."""
        n = 4
        close = np.array([100.0, 100.0, 99.0, 100.0], dtype=np.float64)
        low = np.array([99.0, 99.0, 89.0, 100.0], dtype=np.float64)
        high = np.array([101.0, 101.0, 100.0, 101.0], dtype=np.float64)
        pos = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        stop_s = np.array([math.nan, 90.0, math.nan, math.nan], dtype=np.float64)
        tp_s = np.full(n, math.nan, dtype=np.float64)
        forward = np.array(
            [
                math.log(close[1]) - math.log(close[0]),
                math.log(close[2]) - math.log(close[1]),
                math.log(close[3]) - math.log(close[2]),
                0.0,
            ],
            dtype=np.float64,
        )
        adjust_forward_returns_for_rr_exit_fills(
            forward, pos, low, high, close, stop_s, tp_s
        )
        want = math.log(90.0) - math.log(close[1])
        self.assertAlmostEqual(float(forward[1]), want)


if __name__ == "__main__":
    unittest.main()
