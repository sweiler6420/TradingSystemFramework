# This is trash but a good simple example of how to set the profit return of a strategy to be used

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet('data_stuff/BTCUSD1min.pq')
df.index = df.index.astype('datetime64[s]')

# Sample data for plotting (every 60th point = hourly for visualization)
plot_sample = df.iloc[::60].copy()  # Take every 60th row (hourly data)

fast_ma = df['close'].rolling(10).mean()
slow_ma = df['close'].rolling(50).mean()

# Signal/position vector. The position at each bar
df['signal'] = np.where(fast_ma > slow_ma, 1, 0)

df['return'] = np.log(df['close']).diff().shift(-1)
df['strategy_return'] = df['signal'] * df['return']

r = df['strategy_return']
profit_factor = r[r>0].sum() / r[r<0].abs().sum()
sharpe_ratio = r.mean() / r.std()

print(f"Profit Factor: {profit_factor:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

# Calculate moving averages for the sampled data for plotting
plot_fast_ma = plot_sample['close'].rolling(10).mean()
plot_slow_ma = plot_sample['close'].rolling(50).mean()
plot_signal = np.where(plot_fast_ma > plot_slow_ma, 1, 0)

# Create single plot with price, moving averages, and signals combined
print("Creating combined price and signal plot...")

plt.figure(figsize=(15, 8))

# Plot price and moving averages
plt.plot(plot_sample.index, plot_sample['close'], label='BTC Close Price', linewidth=1, alpha=0.8)
# plt.plot(plot_sample.index, plot_fast_ma, label='Fast MA (10)', linewidth=1.5, color='orange')
# plt.plot(plot_sample.index, plot_slow_ma, label='Slow MA (30)', linewidth=1.5, color='purple')

# Add buy/sell signals only when signal changes (flips)
signal_changes = np.diff(plot_signal, prepend=plot_signal[0]) != 0
change_indices = np.where(signal_changes)[0]

# Plot signal changes only
for idx in change_indices:
    if idx < len(plot_sample):
        signal_value = plot_signal[idx]
        price = plot_sample.iloc[idx]['close']
        time = plot_sample.index[idx]
        
        if signal_value == 1:  # Signal flipped to BUY
            plt.scatter(time, price, color='green', marker='^', s=100, alpha=0.9, 
                       label='Buy Signal' if idx == change_indices[0] else "", zorder=5)
        else:  # Signal flipped to SELL
            plt.scatter(time, price, color='red', marker='v', s=100, alpha=0.9, 
                       label='Sell Signal' if idx == change_indices[0] else "", zorder=5)

plt.title('BTC Moving Average Crossover Strategy with Buy/Sell Signals', fontsize=14, fontweight='bold')
plt.ylabel('Price (USD)')
plt.xlabel('Date')
plt.legend()
plt.grid(True, alpha=0.3)

# Add performance metrics
plt.text(0.02, 0.98, f'Profit Factor: {profit_factor:.4f}\nSharpe Ratio: {sharpe_ratio:.4f}', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
print("Showing plot...")
plt.show()
print("Plot displayed successfully!")
