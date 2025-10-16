"""
Data Handler
============

Handles market data loading, preprocessing, and validation.
Designed primarily for crypto markets (24/7 trading).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple


class DataHandler:
    """
    Handles market data loading, preprocessing, and validation.
    Designed primarily for crypto markets (24/7 trading).
    """
    
    def __init__(self, data_path: str, asset_type: str = "crypto"):
        self.data_path = data_path
        self.asset_type = asset_type
        self.data = None
        self.features = {}
        
    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load market data from file"""
        if self.data_path.endswith('.pq'):
            self.data = pd.read_parquet(self.data_path)
        elif self.data_path.endswith('.csv'):
            self.data = pd.read_csv(self.data_path)
        else:
            raise ValueError("Unsupported file format. Use .pq or .csv")
            
        # Convert timestamp to datetime if needed
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='s')
            self.data.set_index('timestamp', inplace=True)
            
        # Standardize column names
        self.data.columns = [col.lower() for col in self.data.columns]
        
        # Validate required columns for crypto data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return self.data
    
    def filter_date_range(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Filter data by date range"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        mask = (self.data.index.year >= start_year) & (self.data.index.year < end_year)
        self.data = self.data[mask]
        return self.data
    
    def add_features(self, name: str, values: pd.Series):
        """Add calculated features to the dataset"""
        self.features[name] = values
        self.data[name] = values
        
    def get_data(self) -> pd.DataFrame:
        """Get the processed data"""
        return self.data
