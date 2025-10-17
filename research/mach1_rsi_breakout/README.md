# Mach1 RSI Breakout Research

**Project:** mach1_rsi_breakout  
**Description:** RSI breakout strategy with oversold/overbought levels  
**Created:** 2025-10-17

## Strategy Overview

This project implements an RSI breakout strategy that:

- **Enters long positions** when RSI breaks above oversold levels (coming out of oversold)
- **Exits long positions** when RSI breaks below overbought levels (coming out of overbought)
- **Uses long-only approach** for cleaner signal interpretation

## Strategy Logic

### Entry Signal
- RSI was previously ≤ oversold level (30)
- RSI now breaks above oversold level
- Enter long position

### Exit Signal  
- RSI was previously ≥ overbought level (70)
- RSI now breaks below overbought level
- Exit long position

## Parameters

- **RSI Period:** 14 (default)
- **Oversold Level:** 30
- **Overbought Level:** 70
- **Strategy Type:** Long-only

## Usage

```bash
cd research/mach1_rsi_breakout
python main.py
```

## Results

Results will be saved to:
- `results/` - Performance metrics and metadata
- `plots/` - Interactive visualization charts
- `README.md` - This documentation

## Key Differences from Mean Reversion

Unlike traditional RSI mean reversion strategies that:
- Buy when RSI < oversold
- Sell when RSI > overbought

This breakout strategy:
- **Waits for breakout** from oversold before entering
- **Waits for breakout** from overbought before exiting
- **Reduces false signals** by requiring momentum confirmation

## Project Structure

- `data/` - Raw data and processed datasets
- `results/` - Test results, performance metrics, and statistics
- `plots/` - Interactive graphs and visualizations
- `notes/` - Research notes, observations, and findings
- `strategies/` - Strategy implementations specific to this research
- `tests/` - Test scripts and configurations
- `archive/` - Archived results and old versions

## Research Tests

### 1. In-Sample Excellence Test
- **Purpose:** Proof of concept validation
- **Description:** Test strategy performance on historical data
- **Status:** [ ] Not Started / [ ] In Progress / [ ] Completed

### 2. In-Sample Permutation Test
- **Purpose:** Statistical significance validation
- **Description:** Monte Carlo permutation test to validate results
- **Status:** [ ] Not Started / [ ] In Progress / [ ] Completed

### 3. Walk Forward Test
- **Purpose:** Out-of-sample validation
- **Description:** Rolling window validation
- **Status:** [ ] Not Started / [ ] In Progress / [ ] Completed

### 4. Walk Forward Permutation Test
- **Purpose:** Out-of-sample statistical validation
- **Description:** Monte Carlo permutation test on walk-forward results
- **Status:** [ ] Not Started / [ ] In Progress / [ ] Completed

## Key Findings

*To be updated as research progresses...*

## Next Steps

*To be updated as research progresses...*