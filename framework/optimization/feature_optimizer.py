"""
Feature Optimizers
==================

Optimizers specifically designed for individual features.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from framework.features.base_feature import BaseFeature


class FeatureOptimizer(ABC):
    """
    Abstract base class for feature optimization.
    
    Feature optimizers focus on optimizing parameters for individual features,
    making them reusable across different strategies.
    """
    
    @abstractmethod
    def optimize_feature_params(self, data: pd.DataFrame, feature: BaseFeature, **kwargs) -> Dict[str, Any]:
        """
        Optimize parameters for a specific feature.
        
        Args:
            data: Market data
            feature: Feature to optimize
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary with optimized parameters and performance metrics
        """
        pass
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor from returns"""
        winning_trades = returns[returns > 0].sum()
        losing_trades = returns[returns < 0].abs().sum()
        return winning_trades / losing_trades if losing_trades > 0 else 0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio from returns"""
        if returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(252)  # Annualized
