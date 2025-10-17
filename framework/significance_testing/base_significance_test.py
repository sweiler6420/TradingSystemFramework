"""
Base Significance Test Class
===========================

Abstract base class for all significance testing methods.
Significance tests validate that strategy results are statistically significant
and not due to random chance.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class BaseSignificanceTest(ABC):
    """
    Abstract base class for significance testing methods.
    
    Significance tests help validate that trading strategy results are statistically
    significant and not due to random chance. They answer the question:
    "Are these results due to skill or just luck?"
    
    This is different from performance measures, which answer:
    "How good/risky was this strategy?"
    """
    
    def __init__(self, name: str, **params):
        """
        Initialize the significance test.
        
        Args:
            name: Name of the significance test
            **params: Test-specific parameters
        """
        self.name = name
        self.params = params
        self.results = None
        self.is_calculated = False
        
    @abstractmethod
    def test(self, data: pd.DataFrame, strategy_returns: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Perform the significance test.
        
        Args:
            data: Original market data
            strategy_returns: Strategy returns to test
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary with test results including:
            - p_value: Statistical significance
            - is_significant: Boolean indicating if results are significant
            - test_statistic: The calculated test statistic
            - additional test-specific metrics
        """
        pass
    
    def get_results(self, data: pd.DataFrame, strategy_returns: pd.Series, 
                   recalculate: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Get test results, calculating if necessary.
        
        Args:
            data: Original market data
            strategy_returns: Strategy returns to test
            recalculate: Force recalculation even if already calculated
            **kwargs: Additional test parameters
            
        Returns:
            Dictionary with test results
        """
        if not self.is_calculated or recalculate:
            self.results = self.test(data, strategy_returns, **kwargs)
            self.is_calculated = True
            
        return self.results
    
    def get_params(self) -> Dict[str, Any]:
        """Get test parameters"""
        return self.params.copy()
    
    def set_params(self, **params):
        """Set test parameters"""
        self.params.update(params)
        self.is_calculated = False  # Mark for recalculation
    
    def get_test_info(self) -> Dict[str, Any]:
        """
        Get information about the significance test.
        
        Returns:
            Dictionary with test information
        """
        return {
            'name': self.name,
            'params': self.params,
            'is_calculated': self.is_calculated,
            'description': self.__doc__ or f"{self.name} significance test"
        }
    
    def __str__(self) -> str:
        """String representation of the significance test"""
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"
