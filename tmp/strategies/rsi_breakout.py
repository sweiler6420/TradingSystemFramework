# RSI Breakout Strategy Example

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas_ta as ta

df = pd.read_parquet('../data_stuff/BTCUSD1min.pq')
df.index = df.index.astype('datetime64[s]')

df = df[(df.index.year >= 2020) & (df.index.year < 2021)] 

# Use full dataset for plotting

# Calculate RSI using pandas-ta
df['rsi'] = ta.rsi(df['close'], length=14)

# RSI thresholds
sell_threshold = 90  # Overbought - sell signal
buy_threshold = 10   # Oversold - buy signal

# Signal logic: Only long trades
# Enter long when RSI < 20 (oversold)
# Exit long when RSI > 80 (overbought) AND we're currently in a position
# Otherwise hold current position

signal = pd.Series(0, index=df.index)  # Start with no position
in_position = False

for i in range(len(df)):
    rsi_value = df['rsi'].iloc[i]
    
    if not in_position and rsi_value < buy_threshold:
        # Enter long position when oversold
        signal.iloc[i] = 1
        in_position = True
    elif in_position and rsi_value > sell_threshold:
        # Exit long position when overbought
        signal.iloc[i] = 0
        in_position = False
    elif in_position:
        # Hold long position
        signal.iloc[i] = 1
    else:
        # No position
        signal.iloc[i] = 0

df['signal'] = signal

# Calculate returns and strategy performance
df['r'] = np.log(df['close']).diff().shift(-1)
df['strategy_return'] = df['signal'] * df['r']

r = df['strategy_return']
profit_factor = r[r>0].sum() / r[r<0].abs().sum()
sharpe_ratio = r.mean() / r.std()

print(f"Profit Factor: {profit_factor:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Total strategy log return: {df['strategy_return'].sum():.4f}")
print(f"Buy and hold log return: {df['r'].sum():.4f}")

# Use the same RSI and signal data we already calculated

# Create equity curve plot (like Donchian)
print("Creating RSI equity curve...")
plt.figure(figsize=(12, 6))
plt.style.use("dark_background")
df['strategy_return'].cumsum().plot(color='blue', label='RSI Strategy')
df['r'].cumsum().plot(color='gray', label='Buy & Hold', alpha=0.7)
plt.title("RSI Strategy vs Buy & Hold - Cumulative Log Returns")
plt.ylabel('Cumulative Log Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Create detailed RSI strategy plot with shared x-axis for synchronized zooming
print("Creating detailed RSI strategy plot...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1], sharex=True)

# Plot 1: Price and signals (using full dataset)
ax1.plot(df.index, df['close'], label='BTC Close Price', linewidth=0.5, alpha=0.8)

# Add buy/sell signals only when signal changes (flips)
signal_changes = np.diff(df['signal'], prepend=df['signal'].iloc[0]) != 0
change_indices = np.where(signal_changes)[0]

# Plot signal changes only
for idx in change_indices:
    if idx < len(df):
        signal_value = df['signal'].iloc[idx]
        price = df['close'].iloc[idx]
        time = df.index[idx]
        
        if signal_value == 1:  # Signal flipped to BUY (enter long)
            ax1.scatter(time, price, color='green', marker='^', s=30, alpha=0.9, 
                       label='Enter Long' if idx == change_indices[0] else "", zorder=5)
        elif signal_value == 0:  # Signal flipped to SELL (exit long)
            ax1.scatter(time, price, color='red', marker='v', s=30, alpha=0.9, 
                       label='Exit Long' if idx == change_indices[0] else "", zorder=5)

ax1.set_title('BTC RSI Strategy with Buy/Sell Signals', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price (USD)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: RSI (using full dataset)
ax2.plot(df.index, df['rsi'], label='RSI (14)', linewidth=0.5, color='purple')
ax2.axhline(y=sell_threshold, color='red', linestyle='--', alpha=0.7, label=f'Overbought ({sell_threshold})')
ax2.axhline(y=buy_threshold, color='green', linestyle='--', alpha=0.7, label=f'Oversold ({buy_threshold})')
ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)

ax2.set_title('RSI Indicator')
ax2.set_ylabel('RSI')
ax2.set_xlabel('Date')
ax2.set_ylim(0, 100)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add performance metrics
fig.text(0.02, 0.02, f'Profit Factor: {profit_factor:.4f} | Sharpe Ratio: {sharpe_ratio:.4f}', 
         fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
print("Showing plot...")
plt.show()
print("Plot displayed successfully!")
