"""
Suite configuration for mach5_4hr_range_ep2
===================================

Drives validation stages (in-sample excellence, permutation, walk-forward, �).
"""

# Test Configuration (see research/research_runner.py)
TEST_CONFIG = {
    'insample_excellence': {
        'enabled': True,
        'description': 'Proof of concept validation',
        'symbols': ['X:BTCUSD'],
        'interval': '5m',
        'start': '2024-03-01',
        'end': '2026-03-01',
        'provider': 'massive',
        'session_policy': 'CRYPTO_UTC_24H',
        'strategy': 'strategies.mach5_4hr_range_ep2_strategy:Mach54HrRangeEp2Strategy',
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