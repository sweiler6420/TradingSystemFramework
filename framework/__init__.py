"""
Trading Strategy Framework
=========================

A comprehensive OOP framework for developing and testing trading strategies
with three core components:
1. Strategy Idea (indicators, models, features)
2. Development Data (market data handling)
3. Optimizer (parameter selection, model training)

Plus performance measures and Monte Carlo testing.
"""

from framework.strategies import BaseStrategy, Optimizer
from framework.data_handling import DataHandler
from framework.performance import BaseMeasure
from framework.backtest import StrategyBacktest

from framework.strategies.rsi_breakout_strategy import RSIBreakoutStrategy, RSIBreakoutOptimizer
from framework.strategies.donchian_breakout_strategy import DonchianBreakoutStrategy, DonchianBreakoutOptimizer
from framework.features import BaseFeature, RSIFeature, DonchianFeature

__version__ = "1.0.0"
__author__ = "Stephen Weiler"

__all__ = [
    'BaseStrategy',
    'DataHandler',
    'Optimizer', 
    'BaseMeasure',
    'StrategyBacktest',
    'RSIBreakoutStrategy',
    'RSIBreakoutOptimizer',
    'DonchianBreakoutStrategy',
    'DonchianBreakoutOptimizer',
    'BaseFeature',
    'RSIFeature',
    'DonchianFeature'
]
