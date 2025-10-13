# Monte Carlo Permutation Tests for Trading Strategies

This repository contains implementations of Monte Carlo permutation tests for various trading strategies, based on the original work by [neurotrader888](https://github.com/neurotrader888/mcpt).

## Overview

Monte Carlo permutation tests are used to validate the statistical significance of trading strategy performance by comparing actual results against randomly permuted data.

## Files

- `bar_permute.py` - Bar permutation testing utilities
- `donchian.py` - Donchian channel strategy implementation
- `insample_donchian_mcpt.py` - In-sample Monte Carlo permutation test for Donchian strategy
- `insample_tree_mcpt.py` - In-sample Monte Carlo permutation test for tree-based strategy
- `moving_average.py` - Moving average strategy implementation
- `tree_strat.py` - Tree-based trading strategy
- `walkforward_donchian_mcpt.py` - Walk-forward Monte Carlo permutation test for Donchian strategy

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd mcpt
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Each script can be run independently. For example:

```bash
python donchian.py
python insample_donchian_mcpt.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original implementation by [neurotrader888](https://github.com/neurotrader888/mcpt)
- MIT License
