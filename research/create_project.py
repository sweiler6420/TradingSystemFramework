"""
Research Project Entry Script
============================

Creates a new research project directory with standardized structure.
"""

import os
import sys
from datetime import datetime
import argparse


def create_research_project(project_name: str, description: str = "") -> str:
    """
    Create a new research project directory with standardized structure.
    
    Args:
        project_name: Name of the research project (e.g., "mach_1", "rsi_mean_reversion")
        description: Optional description of the research project
        
    Returns:
        str: Path to the created project directory
    """
    
    # Clean project name (replace spaces with underscores, lowercase)
    clean_name = project_name.lower().replace(" ", "_").replace("-", "_")
    
    # Create project directory
    project_dir = os.path.join("research", clean_name)
    os.makedirs(project_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = [
        "data",           # Raw data and processed datasets
        "results",        # Test results, performance metrics
        "plots",          # Interactive graphs and visualizations
        "notes",          # Research notes and observations
        "strategies",     # Strategy implementations specific to this research
        "tests",          # Test scripts and configurations
        "archive"          # Archived results and old versions
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(project_dir, subdir), exist_ok=True)
    
    # Create project README
    readme_content = f"""# {project_name.title()}

**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Description:** {description}

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
"""
    
    with open(os.path.join(project_dir, "README.md"), "w") as f:
        f.write(readme_content)
    
    # Create main research script template
    main_script_content = f'''"""
{project_name.title()} Research Script
====================================

Main research script for {project_name}.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Framework imports
from framework import (
    DataHandler, SignalBasedStrategy, SignalBasedOptimizer,
    RSIFeature, DonchianFeature, PositionState, SignalChange
)
from framework.performance import (
    ProfitFactorMeasure, SharpeRatioMeasure, SortinoRatioMeasure,
    MaxDrawdownMeasure, TotalReturnMeasure, WinRateMeasure
)
from framework.significance_testing import MonteCarloSignificanceTest

# Import the standardized test
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tests.insample_excellence_test import InSampleExcellenceTest

# Import project-specific strategy
from strategies.{clean_name}_strategy import {clean_name.title().replace("_", "")}Strategy


def run_insample_excellence_test():
    """Run in-sample excellence test (proof of concept)"""
    print("=== {project_name.upper()} RESEARCH - IN-SAMPLE EXCELLENCE TEST ===")
    print(f"Started: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
    
    # Load data
    data_handler = DataHandler('framework/data/BTCUSD1hour.pq')
    data_handler.load_data()
    data_handler.filter_date_range(2023, 2024)
    data = data_handler.get_data()
    
    print(f"Data loaded: {{data.shape[0]}} rows from {{data.index[0]}} to {{data.index[-1]}}")
    
    # Create strategy
    strategy = {clean_name.title().replace("_", "")}Strategy()
    
    # Initialize the standardized test
    test = InSampleExcellenceTest(os.path.dirname(__file__))
    
    # Run the test
    test_metadata = test.run_test(strategy, data_handler, "insample_excellence")
    
    # Create plots
    signal_result = strategy.generate_signals()
    test.create_performance_plots(data, signal_result, test_metadata['performance_results'])
    
    # Generate report
    test.generate_test_report(test_metadata)
    
    print(f"\\n=== {project_name.upper()} RESEARCH COMPLETED ===")
    print("Check the following directories for results:")
    print("- results/ - Performance metrics and metadata")
    print("- plots/ - Visualization charts")
    print("- README.md - Project documentation")
    
    return test_metadata


def main():
    """Main research function"""
    print("Starting {project_name} research...")
    
    # Run in-sample excellence test
    results = run_insample_excellence_test()
    
    print(f"\\n{project_name} research completed!")
    print("Check the results/ and plots/ directories for outputs.")


if __name__ == "__main__":
    main()
'''
    
    with open(os.path.join(project_dir, "main.py"), "w") as f:
        f.write(main_script_content)
    
    # Create strategy file template
    strategy_content = f'''"""
{clean_name.title().replace("_", "")}Strategy - {project_name.title()} Strategy
=======================================================

Strategy implementation for {project_name} research project.
"""

import pandas as pd
import numpy as np
from framework import SignalBasedStrategy, RSIFeature


class {clean_name.title().replace("_", "")}Strategy(SignalBasedStrategy):
    """Strategy implementation for {project_name} research"""
    
    def __init__(self, **kwargs):
        super().__init__("{project_name.title()}", long_only=kwargs.get('long_only', False))
        # Initialize your strategy parameters here
        pass
    
    def generate_raw_signals(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate raw trading signals"""
        # Implement your strategy logic here
        signals = pd.Series(0, index=data.index)
        return signals
'''
    
    with open(os.path.join(project_dir, "strategies", f"{clean_name}_strategy.py"), "w") as f:
        f.write(strategy_content)
    
    # Create test configuration file
    test_config_content = f'''"""
Test Configuration for {project_name}
===================================

Configuration settings for all research tests.
"""

# Data Configuration
DATA_CONFIG = {{
    'data_file': 'framework/data/BTCUSD1hour.pq',
    'start_year': 2023,
    'end_year': 2024,
    'insample_start': '2023-01-01',
    'insample_end': '2023-06-30',
    'outsample_start': '2023-07-01',
    'outsample_end': '2023-12-31'
}}

# Strategy Configuration
STRATEGY_CONFIG = {{
    'long_only': False,
    'initial_capital': 10000,
    'commission': 0.001,  # 0.1% commission
    'slippage': 0.0005    # 0.05% slippage
}}

# Test Configuration
TEST_CONFIG = {{
    'insample_excellence': {{
        'enabled': True,
        'description': 'Proof of concept validation'
    }},
    'insample_permutation': {{
        'enabled': False,
        'n_permutations': 1000,
        'description': 'Statistical significance validation'
    }},
    'walk_forward': {{
        'enabled': False,
        'window_size': 252,  # 1 year
        'step_size': 21,     # 1 month
        'description': 'Out-of-sample validation'
    }},
    'walk_forward_permutation': {{
        'enabled': False,
        'n_permutations': 1000,
        'description': 'Out-of-sample statistical validation'
    }}
}}

# Performance Measures
PERFORMANCE_MEASURES = [
    'profit_factor',
    'sharpe_ratio', 
    'sortino_ratio',
    'max_drawdown',
    'total_return',
    'win_rate'
]
'''
    
    with open(os.path.join(project_dir, "tests", "config.py"), "w") as f:
        f.write(test_config_content)
    
    print(f"Research project '{project_name}' created successfully!")
    print(f"Project directory: {project_dir}")
    print(f"Main script: {project_dir}/main.py")
    print(f"README: {project_dir}/README.md")
    print(f"Config: {project_dir}/tests/config.py")
    print(f"\nTo start research, run:")
    print(f"   cd {project_dir}")
    print(f"   python main.py")
    
    return project_dir


def main():
    """Command line interface for creating research projects"""
    parser = argparse.ArgumentParser(description='Create a new research project')
    parser.add_argument('name', help='Name of the research project')
    parser.add_argument('-d', '--description', default='', help='Description of the research project')
    
    args = parser.parse_args()
    
    create_research_project(args.name, args.description)


if __name__ == "__main__":
    main()
