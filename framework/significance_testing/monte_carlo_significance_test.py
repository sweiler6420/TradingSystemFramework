"""
Monte Carlo Significance Test
==============================

Tests whether strategy returns are significantly different from random
by permuting the order of returns and comparing performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_significance_test import BaseSignificanceTest


class MonteCarloSignificanceTest(BaseSignificanceTest):
    """
    Monte Carlo Significance Test for strategy validation.
    
    This test validates that strategy returns are statistically significant
    by comparing them to randomly permuted versions of the same returns.
    
    The null hypothesis is that the strategy's performance is due to random chance.
    We reject this hypothesis if the strategy's performance is significantly
    better than random permutations.
    
    This test helps answer: "Are these results due to skill or just luck?"
    """
    
    def __init__(self, n_permutations: int = 1000, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo Significance Test.
        
        Args:
            n_permutations: Number of random permutations to generate
            random_seed: Random seed for reproducibility
        """
        super().__init__(
            name="Monte Carlo Significance Test",
            n_permutations=n_permutations,
            random_seed=random_seed
        )
        self.n_permutations = n_permutations
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def test(self, data: pd.DataFrame, strategy_returns: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Perform Monte Carlo significance test.
        
        Args:
            data: Original market data (not used in this test)
            strategy_returns: Strategy returns to test
            **kwargs: Additional parameters
                - metric: Performance metric to test ('sharpe', 'profit_factor', 'total_return')
                - confidence_level: Confidence level for significance (default 0.05)
        
        Returns:
            Dictionary with test results
        """
        metric = kwargs.get('metric', 'sharpe')
        confidence_level = kwargs.get('confidence_level', 0.05)
        
        # Calculate the actual strategy performance metric
        actual_metric = self._calculate_metric(strategy_returns, metric)
        
        # Generate random permutations and calculate their metrics
        permuted_metrics = []
        
        for _ in range(self.n_permutations):
            # Create random permutation of returns
            permuted_returns = strategy_returns.sample(frac=1.0).reset_index(drop=True)
            permuted_returns.index = strategy_returns.index  # Restore original index
            
            # Calculate metric for this permutation
            permuted_metric = self._calculate_metric(permuted_returns, metric)
            permuted_metrics.append(permuted_metric)
        
        # Calculate p-value
        permuted_metrics = np.array(permuted_metrics)
        
        if metric in ['sharpe', 'profit_factor', 'total_return']:
            # For metrics where higher is better
            p_value = np.mean(permuted_metrics >= actual_metric)
        else:
            # For metrics where lower is better (like drawdown)
            p_value = np.mean(permuted_metrics <= actual_metric)
        
        # Determine significance
        is_significant = p_value < confidence_level
        
        # Calculate additional statistics
        mean_random_metric = np.mean(permuted_metrics)
        std_random_metric = np.std(permuted_metrics)
        z_score = (actual_metric - mean_random_metric) / std_random_metric if std_random_metric > 0 else 0
        
        return {
            'p_value': p_value,
            'is_significant': is_significant,
            'actual_metric': actual_metric,
            'mean_random_metric': mean_random_metric,
            'std_random_metric': std_random_metric,
            'z_score': z_score,
            'confidence_level': confidence_level,
            'n_permutations': self.n_permutations,
            'metric_tested': metric,
            'test_name': self.name
        }
    
    def _calculate_metric(self, returns: pd.Series, metric: str) -> float:
        """
        Calculate the specified performance metric.
        
        Args:
            returns: Returns series
            metric: Metric to calculate
            
        Returns:
            Calculated metric value
        """
        if metric == 'sharpe':
            if returns.std() == 0:
                return 0
            return returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        
        elif metric == 'profit_factor':
            winning_trades = returns[returns > 0].sum()
            losing_trades = returns[returns < 0].abs().sum()
            return winning_trades / losing_trades if losing_trades > 0 else 0
        
        elif metric == 'total_return':
            return returns.sum()
        
        elif metric == 'max_drawdown':
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def get_significance_summary(self, data: pd.DataFrame, strategy_returns: pd.Series, 
                               **kwargs) -> str:
        """
        Get a human-readable summary of the test results.
        
        Args:
            data: Original market data
            strategy_returns: Strategy returns to test
            **kwargs: Additional test parameters
            
        Returns:
            String summary of test results
        """
        results = self.get_results(data, strategy_returns, **kwargs)
        
        significance_text = "significant" if results['is_significant'] else "not significant"
        
        summary = f"""
Monte Carlo Significance Test Results:
=====================================
Test: {results['test_name']}
Metric Tested: {results['metric_tested']}
P-value: {results['p_value']:.6f}
Result: {significance_text} (α = {results['confidence_level']})

Actual Performance: {results['actual_metric']:.6f}
Random Performance: {results['mean_random_metric']:.6f} ± {results['std_random_metric']:.6f}
Z-score: {results['z_score']:.3f}

Interpretation: {'Strategy results are statistically significant and likely due to skill rather than random chance.' if results['is_significant'] else 'Strategy results are not statistically significant and may be due to random chance.'}
        """.strip()
        
        return summary
