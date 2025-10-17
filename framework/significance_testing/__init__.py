"""
Significance Testing Module
==========================

Contains statistical tests to validate that strategy results are statistically significant
and not due to random chance. These tests help prove that our trading strategies have
genuine edge rather than just luck.
"""

from .base_significance_test import BaseSignificanceTest
from .monte_carlo_significance_test import MonteCarloSignificanceTest

__all__ = [
    'BaseSignificanceTest',
    'MonteCarloSignificanceTest'
]
