"""
Trading Strategy Framework
=========================

A comprehensive OOP framework for developing and testing trading strategies
with two core components:
1. Strategy Idea (indicators, models, features)
2. Development Data (market data handling)

Plus performance measures and significance testing.
"""

from framework.strategies import BaseStrategy, SignalBasedStrategy
from framework.data_handling import DataHandler
from framework.performance import BaseMeasure
from framework.significance_testing import BaseSignificanceTest
from framework.backtest import StrategyBacktest
from framework.features import BaseFeature, RSIFeature, DonchianFeature
from framework.signals import PositionState, SignalChange, SignalResult, SignalManager

__version__ = "1.0.0"
__author__ = "Stephen Weiler"

__all__ = [
    'BaseStrategy',
    'DataHandler',
    'BaseMeasure',
    'BaseSignificanceTest',
    'StrategyBacktest',
    'SignalBasedStrategy',
    'BaseFeature',
    'RSIFeature',
    'DonchianFeature',
    'PositionState',
    'SignalChange',
    'SignalResult',
    'SignalManager'
]
