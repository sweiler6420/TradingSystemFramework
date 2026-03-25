"""
Research Project Entry Script
============================

Creates a new research project directory with standardized structure.
"""

import os
import re
import sys
from datetime import datetime
import argparse
import textwrap

# Add the parent directory to the path to import version_manager
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from research.version_manager import VersionManager


def create_research_project(project_name: str, description: str = "") -> str:
    """
    Create a new research project directory with standardized structure.
    
    Args:
        project_name: Required project title to append to auto-generated mach number (e.g., "rsi_test" becomes "mach3_rsi_test")
        description: Optional description of the research project
        
    Returns:
        str: Path to the created project directory
    """
    
    if not project_name:
        raise ValueError("Project title is required. Use: python research/create_project.py 'your_title'")
    
    # Get the research directory
    research_dir = os.path.dirname(__file__)
    
    # Initialize version manager for project naming
    version_manager = VersionManager(research_dir)
    
    # Generate base project name
    base_name = version_manager.get_next_project_name("mach")
    
    # Append custom title to the base name
    clean_title = project_name.lower().replace(" ", "_").replace("-", "_")
    project_name = f"{base_name}_{clean_title}"
    
    # Clean project name (replace spaces with underscores, lowercase)
    clean_name = project_name.lower().replace(" ", "_").replace("-", "_")
    
    # Create project directory
    project_dir = os.path.join("research", clean_name)
    os.makedirs(project_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = [
        "data",           # Raw data and processed datasets
        "results",        # Versioned run folders (V0001/, …) with metrics, reports, HTML plots
        "notes",          # Research notes and observations
        "strategies",     # Strategy implementations specific to this research
        "configs",        # Suite / run configuration (TEST_CONFIG, etc.)
        "archive",        # Archived results and old versions
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(project_dir, subdir), exist_ok=True)
    
    # Create .gitkeep files for empty directories that should be tracked
    gitkeep_dirs = ["data", "results", "notes", "archive"]
    for gitkeep_dir in gitkeep_dirs:
        gitkeep_file = os.path.join(project_dir, gitkeep_dir, ".gitkeep")
        with open(gitkeep_file, 'w') as f:
            f.write(f"# This file ensures the {gitkeep_dir} directory is tracked by git\n")
            f.write("# even when it's empty. Remove this file if you want to ignore\n")
            f.write("# the entire directory.\n")
    
    # Create project README
    readme_content = textwrap.dedent(f"""
        # {project_name.title()}

        **Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        **Description:** {description}

        ## Project Structure

        - `data/` - Raw data and processed datasets
        - `results/` - Versioned run directories (`V0001/`, …) with CSV/JSON, markdown reports, and interactive HTML plots
        - `notes/` - Research notes, observations, and findings
        - `strategies/` - Strategy implementations specific to this research
        - `configs/` - Suite and validation run configuration (`config.py`)
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
    """).strip()
    
    with open(os.path.join(project_dir, "README.md"), "w") as f:
        f.write(readme_content)

    # Thin entry: matches research/mach4_ema_band_ep1/main.py (orchestration in research_runner + configs/config)
    _title_line = f"{project_name.replace('_', ' ').title()} — research entry (optional)"
    _title_under = "=" * len(_title_line)
    _mach_m = re.match(r"^mach(\d+)_", clean_name)
    if _mach_m:
        _n = _mach_m.group(1)
        _preferred = (
            f"Preferred: from repo root run ``python research/run.py {clean_name}``\n"
            f"(or ``{_n}`` / ``mach{_n}`` if unambiguous). That uses :mod:`research.research_runner` and\n"
            f"``configs/config.py`` — same behavior as this file."
        )
    else:
        _preferred = (
            f"Preferred: from repo root run ``python research/run.py {clean_name}``.\n"
            f"That uses :mod:`research.research_runner` and\n"
            f"``configs/config.py`` — same behavior as this file."
        )

    main_script_content = textwrap.dedent(f'''
        """
        {_title_line}
        {_title_under}

        {_preferred}

        Set ``MASSIVE_API_KEY`` for live fetches when the Parquet cache is missing.
        """

        from __future__ import annotations

        import os
        import sys


        def main() -> None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            project_dir = os.path.dirname(os.path.abspath(__file__))
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            if sys.path[0] != project_dir:
                sys.path.insert(0, project_dir)

            from research.research_runner import run_cli

            run_cli(project_dir, repo_root=repo_root)


        if __name__ == "__main__":
            main()
    ''').strip()
    
    with open(os.path.join(project_dir, "main.py"), "w") as f:
        f.write(main_script_content)
    
    # Create strategy file template
    strategy_content = textwrap.dedent(f"""
        \"\"\"
        {clean_name.title().replace("_", "")}Strategy - {project_name.title()} Strategy
        =======================================================

        Strategy implementation for {project_name} research project.
        \"\"\"

        import polars as pl
        import numpy as np
        from framework import SignalBasedStrategy, SignalChange


        class {clean_name.title().replace("_", "")}Strategy(SignalBasedStrategy):
            \"\"\"Strategy implementation for {project_name} research\"\"\"
            
            def __init__(self, data: pl.DataFrame, **kwargs):
                super().__init__("{project_name.title()}")
                self._data = data
                # Initialize your strategy parameters here
            
            def generate_raw_signals(self, data: pl.DataFrame, **kwargs) -> pl.Series:
                \"\"\"Generate raw trading signals\"\"\"
                # Implement your strategy logic here
                signals = pl.Series([SignalChange.NO_CHANGE] * len(data))
                return signals
    """).strip()
    
    with open(os.path.join(project_dir, "strategies", f"{clean_name}_strategy.py"), "w") as f:
        f.write(strategy_content)

    configs_init = os.path.join(project_dir, "configs", "__init__.py")
    if not os.path.isfile(configs_init):
        with open(configs_init, "w") as f:
            f.write('"""Suite and validation configuration for this research project."""\n')
    
    # Create suite / validation configuration (configs/config.py)
    test_config_content = textwrap.dedent(f"""
        \"\"\"
        Suite configuration for {project_name}
        ===================================

        Drives validation stages (in-sample excellence, permutation, walk-forward, …).
        \"\"\"

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

        # Test Configuration (see research/research_runner.py)
        TEST_CONFIG = {{
            'insample_excellence': {{
                'enabled': True,
                'description': 'Proof of concept validation',
                'symbols': ['C:EURUSD'],
                'interval': '1h',
                'start': '2024-01-01',
                'end': '2024-06-30',
                'provider': 'massive',
                'session_policy': 'CRYPTO_UTC_24H',
                'strategy': 'strategies.{clean_name}_strategy:{clean_name.title().replace("_", "")}Strategy',
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
    """).strip()
    
    with open(os.path.join(project_dir, "configs", "config.py"), "w") as f:
        f.write(test_config_content)
    
    print(f"Research project '{project_name}' created successfully!")
    print(f"Project directory: {project_dir}")
    print(f"Main script: {project_dir}/main.py")
    print(f"README: {project_dir}/README.md")
    print(f"Config: {project_dir}/configs/config.py")
    print(f"\nTo start research (from repo root), run:")
    print(f"   python research/run.py {clean_name}")
    print(f"   # or: cd {project_dir} && python main.py")
    
    return project_dir


def main():
    """Command line interface for creating research projects"""
    parser = argparse.ArgumentParser(
        description='Create a new research project with automatic mach numbering',
        epilog=textwrap.dedent('''
            Examples:
              python research/create_project.py rsi_mean_reversion
              python research/create_project.py --title rsi_mean_reversion
              python research/create_project.py -t donchian_breakout -d "Testing Donchian breakout strategy"
              python research/create_project.py bollinger_bands -d "Bollinger Bands mean reversion test"

            The project name will be automatically prefixed with the next available mach number:
              rsi_test -> mach6_rsi_test
              moving_average -> mach7_moving_average
        ''').strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'title',
        nargs='?',
        default=None,
        metavar='TITLE',
        help='Project title — appended to the next machN_ prefix (same as -t/--title)',
    )
    parser.add_argument(
        '-t',
        '--title',
        dest='title_flag',
        metavar='TITLE',
        default=None,
        help='Project title (alternative to the positional TITLE)',
    )
    parser.add_argument('-d', '--description', default='', help='Optional description of the research project')

    args = parser.parse_args()

    title = args.title if args.title is not None else args.title_flag
    if args.title is not None and args.title_flag is not None and args.title != args.title_flag:
        parser.error('positional TITLE and -t/--title disagree; pass only one title')
    if not title:
        parser.error('a project title is required (positional TITLE or -t/--title)')

    try:
        create_research_project(title, args.description)
    except ValueError as e:
        print(f"Error: {e}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
