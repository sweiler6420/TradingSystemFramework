"""
Data Handler
============

Handles market data loading, preprocessing, and validation.
Designed primarily for crypto markets (24/7 trading).
"""

import polars as pl
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
        
    def load_data(self, **kwargs) -> pl.DataFrame:
        """Load market data from file"""
        if self.data_path.endswith('.pq'):
            self.data = pl.read_parquet(self.data_path)
        elif self.data_path.endswith('.csv'):
            self.data = pl.read_csv(self.data_path)
        else:
            raise ValueError("Unsupported file format. Use .pq or .csv")
            
        # Convert timestamp to datetime if needed
        if 'timestamp' in self.data.columns:
            self.data = self.data.with_columns(
                pl.col('timestamp').cast(pl.Datetime).alias('timestamp')
            )
            self.data = self.data.set_sorted('timestamp')
            
        # Standardize column names
        self.data = self.data.rename({col: col.lower() for col in self.data.columns})
        
        # Validate required columns for crypto data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return self.data
    
    def filter_date_range(self, start_year: int, end_year: int) -> pl.DataFrame:
        """Filter data by date range"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Filter by year range
        if 'timestamp' in self.data.columns:
            self.data = self.data.filter(
                (pl.col('timestamp').dt.year() >= start_year) & 
                (pl.col('timestamp').dt.year() < end_year)
            )
        else:
            # If no timestamp column, assume index is datetime
            raise ValueError("No timestamp column found for date filtering")
            
        return self.data
    
    def add_features(self, name: str, values: pl.Series):
        """Add calculated features to the dataset"""
        self.features[name] = values
        self.data = self.data.with_columns(values.alias(name))
        
    def get_data(self) -> pl.DataFrame:
        """Get the processed data"""
        return self.data
