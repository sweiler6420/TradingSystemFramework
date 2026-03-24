"""
Features Module
==============

Contains feature classes for trading strategies and machine learning models.
Features are modular indicators that can be applied to any strategy.
"""

from framework.features.base_feature import BaseFeature
from framework.features.donchian_feature import DonchianFeature
from framework.features.ema_feature import (
    EmaFeature,
    OHLC_PRICE_COLUMNS,
    validate_ohlc_price_column,
)
from framework.features.macd_feature import MacdFeature
from framework.features.rsi_feature import RSIFeature

__all__ = [
    'BaseFeature',
    'DonchianFeature',
    'EmaFeature',
    'MacdFeature',
    'OHLC_PRICE_COLUMNS',
    'RSIFeature',
    'validate_ohlc_price_column',
]
