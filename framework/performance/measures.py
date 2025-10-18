"""
Performance Measures Base Classes
=================================

Base classes for performance measurement tools.
"""

import polars as pl
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List


class BaseMeasure(ABC):
    """
    Abstract base class for all performance measures.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate(self, data: pl.DataFrame, **kwargs) -> Any:
        """Calculate the performance measure"""
        pass
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.name}"


