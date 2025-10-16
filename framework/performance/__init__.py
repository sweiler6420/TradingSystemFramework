"""
Performance Analysis Module
==========================

Contains performance measurement tools and Monte Carlo testing capabilities.
"""

# Base classes
from framework.performance.measures import BaseMeasure

# Basic performance measures
from framework.performance.returns_measure import ReturnsMeasure
from framework.performance.profit_factor_measure import ProfitFactorMeasure
from framework.performance.sharpe_ratio_measure import SharpeRatioMeasure
from framework.performance.max_drawdown_measure import MaxDrawdownMeasure
from framework.performance.total_return_measure import TotalReturnMeasure
from framework.performance.win_rate_measure import WinRateMeasure
from framework.performance.total_trades_measure import TotalTradesMeasure

# Monte Carlo testing measures
from framework.performance.monte_carlo_measures import MonteCarloPermutationTest

# Advanced risk measures
from framework.performance.calmar_ratio_measure import CalmarRatioMeasure
from framework.performance.sortino_ratio_measure import SortinoRatioMeasure
from framework.performance.var_measure import VaRMeasure
from framework.performance.cvar_measure import CVaRMeasure

__all__ = [
    # Base classes
    'BaseMeasure',
    
    # Basic performance measures
    'ReturnsMeasure',
    'ProfitFactorMeasure',
    'SharpeRatioMeasure',
    'MaxDrawdownMeasure',
    'TotalReturnMeasure',
    'WinRateMeasure',
    'TotalTradesMeasure',
    
    # Monte Carlo testing measures
    'MonteCarloPermutationTest',
    
    # Advanced risk measures
    'CalmarRatioMeasure',
    'SortinoRatioMeasure',
    'VaRMeasure',
    'CVaRMeasure'
]
