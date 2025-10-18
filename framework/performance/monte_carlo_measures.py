"""
Monte Carlo Testing Measures
============================

Monte Carlo permutation testing and related statistical measures.
"""

import polars as pl
import numpy as np
from framework.performance.measures import BaseMeasure
from typing import Dict, Any, Union, List


class MonteCarloPermutationTest(BaseMeasure):
    """
    Monte Carlo permutation test to check for data mining bias.
    """
    
    def __init__(self, n_permutations: int = 1000, significance_level: float = 0.05):
        super().__init__("Monte Carlo Permutation Test")
        self.n_permutations = n_permutations
        self.significance_level = significance_level
    
    def calculate(self, data: pl.DataFrame, strategy_returns: pl.Series, **kwargs) -> Dict[str, Any]:
        """
        Run Monte Carlo permutation test.
        
        Args:
            data: Original market data
            strategy_returns: Strategy returns to test
            **kwargs: Additional parameters for permutation methods
        """
        n_permutations = kwargs.get('n_permutations', self.n_permutations)
        significance_level = kwargs.get('significance_level', self.significance_level)
        
        permutation_returns = []
        
        for i in range(n_permutations):
            # Create permuted data (filter out strategy-specific kwargs)
            permutation_kwargs = {k: v for k, v in kwargs.items() 
                                if k in ['start_index', 'seed']}
            permuted_data = self.get_permutation(data, **permutation_kwargs)
            
            # Recalculate strategy on permuted data
            # Note: This is a simplified version - you'd need to re-run your strategy
            permuted_returns = np.random.permutation(strategy_returns.to_numpy())
            permutation_returns.append(permuted_returns.sum())
        
        # Calculate statistics
        actual_return = strategy_returns.sum()
        permutation_returns = np.array(permutation_returns)
        
        p_value = np.mean(permutation_returns >= actual_return)
        
        return {
            'actual_return': actual_return,
            'permutation_mean': permutation_returns.mean(),
            'permutation_std': permutation_returns.std(),
            'p_value': p_value,
            'is_significant': p_value < significance_level,
            'permutation_returns': permutation_returns,
            'n_permutations': n_permutations,
            'significance_level': significance_level
        }
    
    @staticmethod
    def get_permutation(ohlc: Union[pl.DataFrame, List[pl.DataFrame]], 
                       start_index: int = 0, seed: int = None) -> Union[pl.DataFrame, List[pl.DataFrame]]:
        """
        Create a permutation of OHLC data while preserving some statistical properties.
        
        This is based on the bar_permute.py logic but adapted for the framework.
        """
        np.random.seed(seed)

        if isinstance(ohlc, list):
            time_index = ohlc[0].index
            for mkt in ohlc:
                assert np.all(time_index == mkt.index), "Indexes do not match"
            n_markets = len(ohlc)
        else:
            n_markets = 1
            time_index = ohlc.index
            ohlc = [ohlc]

        n_bars = len(ohlc[0])

        perm_index = start_index + 1
        perm_n = n_bars - perm_index

        start_bar = np.empty((n_markets, 4))
        relative_open = np.empty((n_markets, perm_n))
        relative_high = np.empty((n_markets, perm_n))
        relative_low = np.empty((n_markets, perm_n))
        relative_close = np.empty((n_markets, perm_n))

        for mkt_i, reg_bars in enumerate(ohlc):
            log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']])

            # Get start bar
            start_bar[mkt_i] = log_bars.iloc[start_index].to_numpy()

            # Open relative to last close
            r_o = (log_bars['open'] - log_bars['close'].shift()).to_numpy()
            
            # Get prices relative to this bars open
            r_h = (log_bars['high'] - log_bars['open']).to_numpy()
            r_l = (log_bars['low'] - log_bars['open']).to_numpy()
            r_c = (log_bars['close'] - log_bars['open']).to_numpy()

            relative_open[mkt_i] = r_o[perm_index:]
            relative_high[mkt_i] = r_h[perm_index:]
            relative_low[mkt_i] = r_l[perm_index:]
            relative_close[mkt_i] = r_c[perm_index:]

        idx = np.arange(perm_n)

        # Shuffle intrabar relative values (high/low/close)
        perm1 = np.random.permutation(idx)
        relative_high = relative_high[:, perm1]
        relative_low = relative_low[:, perm1]
        relative_close = relative_close[:, perm1]

        # Shuffle last close to open (gaps) separately
        perm2 = np.random.permutation(idx)
        relative_open = relative_open[:, perm2]

        # Create permutation from relative prices
        perm_ohlc = []
        for mkt_i, reg_bars in enumerate(ohlc):
            perm_bars = np.zeros((n_bars, 4))

            # Copy over real data before start index 
            log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']]).to_numpy().copy()
            perm_bars[:start_index] = log_bars[:start_index]
            
            # Copy start bar
            perm_bars[start_index] = start_bar[mkt_i]

            for i in range(perm_index, n_bars):
                k = i - perm_index
                perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[mkt_i][k]
                perm_bars[i, 1] = perm_bars[i, 0] + relative_high[mkt_i][k]
                perm_bars[i, 2] = perm_bars[i, 0] + relative_low[mkt_i][k]
                perm_bars[i, 3] = perm_bars[i, 0] + relative_close[mkt_i][k]

            perm_bars = np.exp(perm_bars)
            perm_bars = pl.DataFrame(perm_bars, schema=['open', 'high', 'low', 'close'])

            perm_ohlc.append(perm_bars)

        if n_markets > 1:
            return perm_ohlc
        else:
            return perm_ohlc[0]

    @staticmethod
    def simple_permutation(data: pl.DataFrame, seed: int = None) -> pl.DataFrame:
        """
        Simple random permutation of returns (destroys all temporal dependencies).
        Use with caution as this creates unrealistic market conditions.
        """
        np.random.seed(seed)
        permuted_data = data.copy()
        
        # Permute returns
        returns = np.log(data['close']).diff().shift(-1)
        permuted_returns = np.random.permutation(returns)
        
        # Reconstruct price series from permuted returns
        log_prices = np.log(data['close'].iloc[0]) + permuted_returns.cumsum()
        permuted_data['close'] = np.exp(log_prices)
        
        # Reconstruct OHLC from close prices (simplified)
        permuted_data['open'] = permuted_data['close'].shift(1)
        permuted_data['high'] = permuted_data[['open', 'close']].max(axis=1) * 1.001
        permuted_data['low'] = permuted_data[['open', 'close']].min(axis=1) * 0.999
        
        return permuted_data

    @staticmethod
    def block_permutation(data: pl.DataFrame, block_size: int = 20, seed: int = None) -> pl.DataFrame:
        """
        Block permutation that preserves short-term dependencies within blocks.
        """
        np.random.seed(seed)
        
        # Create blocks
        n_blocks = len(data) // block_size
        blocks = []
        
        for i in range(0, len(data) - block_size + 1, block_size):
            block = data.iloc[i:i + block_size].copy()
            blocks.append(block)
        
        # Permute blocks
        permuted_blocks = np.random.permutation(blocks)
        
        # Reconstruct data
        permuted_data = pl.concat(permuted_blocks)
        
        return permuted_data
