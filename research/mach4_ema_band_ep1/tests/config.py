"""
Test configuration for mach4_ema_band_ep1
=========================================
"""

DATA_CONFIG = {
    # Cache filename uses ``=`` → ``_`` in ``safe_symbol_label`` (see framework.data_sources.cache).
    "data_file": "data/EURUSD_X_1h_*.parquet",
    "start_year": 2024,
    "end_year": 2026,
    "insample_start": "2024-06-01",
    "insample_end": "2024-12-31",
    "outsample_start": "2025-01-01",
    "outsample_end": "2025-12-31",
}

STRATEGY_CONFIG = {
    "long_only": True,
    "initial_capital": 10000,
    "commission": 0.001,
    "slippage": 0.0005,
}

TEST_CONFIG = {
    "insample_excellence": {
        "enabled": True,
        "description": "Proof of concept validation",
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
