"""
Test Configuration for mach1_rsi_breakout
===================================

Configuration settings for all research tests.
"""

# Data Configuration
DATA_CONFIG = {
    'data_file': 'framework/data/BTCUSD1hour.pq',
    'start_year': 2023,
    'end_year': 2024,
    'insample_start': '2023-01-01',
    'insample_end': '2023-06-30',
    'outsample_start': '2023-07-01',
    'outsample_end': '2023-12-31'
}

# Strategy Configuration
STRATEGY_CONFIG = {
    'long_only': False,
    'initial_capital': 10000,
    'commission': 0.001,  # 0.1% commission
    'slippage': 0.0005    # 0.05% slippage
}

# Test Configuration
TEST_CONFIG = {
    'insample_excellence': {
        'enabled': True,
        'description': 'Proof of concept validation'
    },
    'insample_permutation': {
        'enabled': False,
        'n_permutations': 1000,
        'description': 'Statistical significance validation'
    },
    'walk_forward': {
        'enabled': False,
        'window_size': 252,  # 1 year
        'step_size': 21,     # 1 month
        'description': 'Out-of-sample validation'
    },
    'walk_forward_permutation': {
        'enabled': False,
        'n_permutations': 1000,
        'description': 'Out-of-sample statistical validation'
    }
}

# Performance Measures
PERFORMANCE_MEASURES = [
    'profit_factor',
    'sharpe_ratio', 
    'sortino_ratio',
    'max_drawdown',
    'total_return',
    'win_rate'
]
