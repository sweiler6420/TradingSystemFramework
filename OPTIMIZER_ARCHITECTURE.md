# Optimizer Architecture for Feature-Based Trading Framework

## Current Problem

With the new feature-based architecture, we need to rethink how optimizers work. The current approach has one optimizer per strategy, but this creates issues:

1. **Tight Coupling**: Optimizers are tightly bound to specific strategy implementations
2. **Code Duplication**: Similar optimization logic is repeated across strategies
3. **Poor Reusability**: Feature optimizations can't be reused across different strategies
4. **Complexity**: Strategies need to expose internal methods for optimization

## Recommended Architecture: Hybrid Approach

### 1. Feature-Level Optimizers

Each feature should have its own optimizer for its specific parameters:

```python
class FeatureOptimizer(ABC):
    """Abstract base class for feature optimization"""
    
    @abstractmethod
    def optimize_feature_params(self, data: pd.DataFrame, feature: BaseFeature, **kwargs) -> Dict[str, Any]:
        """Optimize parameters for a specific feature"""
        pass

class RSIFeatureOptimizer(FeatureOptimizer):
    """Optimizer specifically for RSI feature parameters"""
    
    def optimize_feature_params(self, data: pd.DataFrame, feature: RSIFeature, **kwargs):
        # Optimize RSI period, overbought/oversold thresholds
        # Return best parameters for this feature
        pass

class DonchianFeatureOptimizer(FeatureOptimizer):
    """Optimizer specifically for Donchian feature parameters"""
    
    def optimize_feature_params(self, data: pd.DataFrame, feature: DonchianFeature, **kwargs):
        # Optimize lookback period, thresholds
        # Return best parameters for this feature
        pass
```

### 2. Strategy-Level Optimizers

Strategies should have optimizers that:
- Use feature optimizers for individual features
- Optimize strategy-specific parameters (position sizing, timing, etc.)
- Coordinate feature combinations

```python
class StrategyOptimizer(ABC):
    """Abstract base class for strategy optimization"""
    
    @abstractmethod
    def optimize_strategy(self, data: pd.DataFrame, strategy: BaseStrategy, **kwargs) -> Dict[str, Any]:
        """Optimize the complete strategy including all features"""
        pass

class RSIBreakoutStrategyOptimizer(StrategyOptimizer):
    """Optimizer for RSI breakout strategy"""
    
    def __init__(self):
        self.rsi_optimizer = RSIFeatureOptimizer()
    
    def optimize_strategy(self, data: pd.DataFrame, strategy: RSIBreakoutStrategy, **kwargs):
        # 1. Optimize RSI feature parameters
        rsi_params = self.rsi_optimizer.optimize_feature_params(
            data, strategy.rsi_feature, **kwargs
        )
        
        # 2. Optimize strategy-specific parameters (if any)
        strategy_params = self._optimize_strategy_params(data, strategy, **kwargs)
        
        # 3. Return combined optimal parameters
        return {**rsi_params, **strategy_params}
```

### 3. Composite Feature Strategies

For strategies using multiple features:

```python
class MultiFeatureStrategyOptimizer(StrategyOptimizer):
    """Optimizer for strategies using multiple features"""
    
    def __init__(self):
        self.feature_optimizers = {
            'rsi': RSIFeatureOptimizer(),
            'donchian': DonchianFeatureOptimizer(),
            'macd': MACDFeatureOptimizer(),  # Future feature
        }
    
    def optimize_strategy(self, data: pd.DataFrame, strategy: MultiFeatureStrategy, **kwargs):
        optimized_params = {}
        
        # Optimize each feature independently
        for feature_name, feature in strategy.features.items():
            if feature_name in self.feature_optimizers:
                feature_params = self.feature_optimizers[feature_name].optimize_feature_params(
                    data, feature, **kwargs
                )
                optimized_params.update(feature_params)
        
        # Optimize feature interactions/weights
        interaction_params = self._optimize_feature_interactions(data, strategy, **kwargs)
        optimized_params.update(interaction_params)
        
        return optimized_params
```

## Benefits of This Approach

### 1. **Modularity**
- Feature optimizers can be reused across different strategies
- Easy to add new features with their own optimizers
- Clear separation of concerns

### 2. **Flexibility**
- Can optimize features independently or together
- Support for different optimization objectives per feature
- Easy to add new optimization techniques

### 3. **Extensibility**
- New features automatically get their own optimizers
- Strategies can easily combine multiple features
- Support for complex multi-feature strategies

### 4. **Testability**
- Feature optimizers can be tested independently
- Strategy optimizers can be tested with mock feature optimizers
- Clear interfaces make testing easier

## Implementation Strategy

### Phase 1: Refactor Current Optimizers
1. Create `FeatureOptimizer` base class
2. Move feature-specific optimization logic to feature optimizers
3. Update strategy optimizers to use feature optimizers

### Phase 2: Enhance Feature Optimizers
1. Add more sophisticated optimization algorithms
2. Support for different optimization objectives
3. Feature interaction optimization

### Phase 3: Advanced Features
1. Multi-objective optimization
2. Real-time parameter adaptation
3. Feature selection optimization

## Example Usage

```python
# Create a strategy with multiple features
strategy = MultiFeatureStrategy()
strategy.add_feature('rsi', RSIFeature())
strategy.add_feature('donchian', DonchianFeature())

# Create optimizer
optimizer = MultiFeatureStrategyOptimizer()

# Optimize the strategy
best_params = optimizer.optimize_strategy(data, strategy)

# Apply optimized parameters
strategy.apply_parameters(best_params)
```

## Migration Path

1. **Immediate**: Keep current optimizers working while we refactor
2. **Short-term**: Implement feature optimizers alongside strategy optimizers
3. **Long-term**: Migrate to pure feature-based optimization

This approach maintains backward compatibility while providing a clear path forward for more sophisticated optimization strategies.
