"""
Base Feature Class
=================

Abstract base class for all features (indicators) that can be used in trading strategies
or machine learning models.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union


class BaseFeature(ABC):
    """
    Abstract base class for all features.
    
    Features are modular indicators that can be applied to any strategy.
    They can be traditional trading indicators (RSI, Donchian, etc.) or
    any other engineered features for machine learning models.
    """
    
    def __init__(self, name: str, **params):
        """
        Initialize the feature.
        
        Args:
            name: Name of the feature
            **params: Feature-specific parameters
        """
        self.name = name
        self.params = params
        self.values = None
        self.is_calculated = False
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the feature values.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with calculated feature values
        """
        pass
    
    def get_values(self, data: pd.DataFrame, recalculate: bool = False) -> pd.Series:
        """
        Get feature values, calculating if necessary.
        
        Args:
            data: DataFrame with OHLCV data
            recalculate: Force recalculation even if already calculated
            
        Returns:
            Series with feature values
        """
        if not self.is_calculated or recalculate:
            self.values = self.calculate(data)
            self.is_calculated = True
            
        return self.values
    
    def get_params(self) -> Dict[str, Any]:
        """Get feature parameters"""
        return self.params.copy()
    
    def set_params(self, **params):
        """Set feature parameters"""
        self.params.update(params)
        self.is_calculated = False  # Mark for recalculation
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the data contains required columns.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in data.columns for col in required_columns)
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about the feature.
        
        Returns:
            Dictionary with feature information
        """
        return {
            'name': self.name,
            'params': self.params,
            'is_calculated': self.is_calculated,
            'description': self.__doc__ or f"{self.name} feature"
        }
    
    def __str__(self) -> str:
        """String representation of the feature"""
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"
