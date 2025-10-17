"""
Strategy Optimizers
===================

Optimizers that coordinate feature optimization and strategy-specific parameters.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from framework.strategies.base_strategy import BaseStrategy
from .feature_optimizer import FeatureOptimizer


class StrategyOptimizer(ABC):
    """
    Abstract base class for strategy optimization.
    
    Strategy optimizers coordinate feature optimization and handle
    strategy-specific parameters and feature interactions.
    """
    
    @abstractmethod
    def optimize_strategy(self, data: pd.DataFrame, strategy: BaseStrategy, **kwargs) -> Dict[str, Any]:
        """
        Optimize the complete strategy including all features.
        
        Args:
            data: Market data
            strategy: Strategy to optimize
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary with optimized parameters and performance metrics
        """
        pass
