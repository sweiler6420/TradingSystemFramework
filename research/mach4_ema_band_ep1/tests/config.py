"""
Test configuration for mach4_ema_band_ep1
=========================================
"""

from __future__ import annotations

# ``insample_excellence`` is read by ``main.py`` when running the in-sample suite.
TEST_CONFIG = {
    "insample_excellence": {
        "enabled": True,
        "description": "Proof of concept validation",
        # Massive tickers (e.g. C:EURUSD forex). One suite run per symbol.
        "symbols": [
            "C:EURUSD",
        ],
        "interval": "1h",
        # Inclusive calendar dates for the analysis window (after load).
        "start": "2024-03-01",
        "end": "2026-03-01",
        # Optional: override Parquet cache path only (must match filename dates in data/).
        # Defaults: same as start/end — second date in the name is ``cache_end``, not end+1 day.
        # "cache_start": "2024-03-01",
        # "cache_end": "2026-03-01",
        # Provider for this project (Massive / Polygon).
        "provider": "massive",
        # ``framework.data_handling.market_session.SessionPolicy`` name for cache + bars.
        "session_policy": "CRYPTO_UTC_24H",
        # Strategy class for ``research.research_runner`` (module under this project : ClassName).
        "strategy": "strategies.ema_band_ep1_strategy:EmaBandEp1Strategy",
    },
    "insample_permutation": {
        "enabled": False,
        "n_permutations": 1000,
        "description": "Statistical significance validation",
    },
    "walk_forward": {
        "enabled": False,
        "window_size": 252,
        "step_size": 21,
        "description": "Out-of-sample validation",
    },
    "walk_forward_permutation": {
        "enabled": False,
        "n_permutations": 1000,
        "description": "Out-of-sample statistical validation",
    },
}

PERFORMANCE_MEASURES = [
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "total_return",
    "win_rate",
]
