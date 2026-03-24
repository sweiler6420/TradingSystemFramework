"""
Data Handler
============

Handles market data loading, preprocessing, and validation.

Use ``SessionPolicy`` so backtests use the same bar universe (e.g. US RTH vs
24h crypto) as your live definition.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from framework.data_handling.market_session import SessionPolicy, apply_session_policy


class DataHandler:
    """
    Handles market data loading, preprocessing, and validation.

    ``asset_type`` selects a default session policy when ``session_policy`` is
    omitted: ``"stock"`` → US regular hours; ``"crypto"`` → 24h UTC (no row drop).
    """

    def __init__(
        self,
        data_path: str,
        asset_type: str = "crypto",
        *,
        session_policy: SessionPolicy | None = None,
        naive_timestamp_tz: str | None = None,
    ):
        self.data_path = data_path
        self.asset_type = asset_type
        if session_policy is None:
            if asset_type == "stock":
                session_policy = SessionPolicy.US_EQUITY_RTH
            else:
                session_policy = SessionPolicy.CRYPTO_UTC_24H
        self.session_policy = session_policy
        # Yahoo intraday (and most vendor) timestamps are naive UTC; override if your
        # file stores naive wall time in another zone (e.g. America/New_York).
        if naive_timestamp_tz is None:
            naive_timestamp_tz = "UTC"
        self.naive_timestamp_tz = naive_timestamp_tz
        self.data = None
        self.features = {}
        
    def load_data(self, **kwargs) -> pl.DataFrame:
        """Load market data from file"""
        path_lower = self.data_path.lower()
        if path_lower.endswith((".pq", ".parquet")):
            self.data = pl.read_parquet(self.data_path)
        elif path_lower.endswith(".csv"):
            self.data = pl.read_csv(self.data_path)
        else:
            raise ValueError("Unsupported file format. Use .parquet, .pq, or .csv")
            
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

        self.data = apply_session_policy(
            self.data,
            self.session_policy,
            naive_timestamp_tz=self.naive_timestamp_tz,
        )

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
