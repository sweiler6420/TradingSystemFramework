"""Errors raised by market data providers."""


class YFinanceDataError(ValueError):
    """
    Raised when yfinance cannot return usable OHLCV for the requested parameters.

    Typical causes: symbol not found / delisted, intraday range outside Yahoo's
    rolling ~730-day window, or repeated empty responses for the requested chunks.
    """
