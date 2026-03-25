"""
In-sample excellence suite
==========================

Proof-of-concept validation: performance measures + Bokeh interactive reports.
"""

import json
import os
import sys
import textwrap
from datetime import datetime
from typing import Any, Dict

import polars as pl

# Framework imports
from framework import DataHandler, SignalBasedStrategy
from framework.performance import (
    MaxDrawdownMeasure,
    ProfitFactorMeasure,
    SharpeRatioMeasure,
    SortinoRatioMeasure,
    TotalReturnMeasure,
    WinRateMeasure,
)

# Project root on path for `research.version_manager`
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from research.version_manager import VersionManager

from .bokeh_interactive_plot_creator import BokehInteractivePlotCreator


class InSampleExcellenceSuite:
    """In-sample excellence suite: metrics, plots, and versioned metadata."""

    def __init__(self, project_dir: str, strategy: SignalBasedStrategy):
        self.project_dir = project_dir
        self.results_dir = os.path.join(project_dir, "results")
        self.strategy = strategy
        self._run_dir: str | None = None

        self.measures = {
            "profit_factor": ProfitFactorMeasure(),
            "sharpe_ratio": SharpeRatioMeasure(),
            "sortino_ratio": SortinoRatioMeasure(),
            "max_drawdown": MaxDrawdownMeasure(),
            "total_return": TotalReturnMeasure(),
            "win_rate": WinRateMeasure(),
        }

        # Run folders: ``results/V0003/`` (version only). Artifacts use ``{test_name}_*`` filenames.
        self.version_manager = VersionManager(self.results_dir)

    def run_test(
        self,
        data_handler: DataHandler,
        test_name: str = "insample_excellence",
        *,
        run_version_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        Run the in-sample excellence suite.

        Args:
            data_handler: Data handler with loaded data
            test_name: Name for saving results (artifact prefix under the run folder)
            run_version_id: If set (e.g. ``V0003``), write into ``results/run_version_id/``
                without allocating a new id — use one id for a batch (multiple symbols / suites).

        Returns:
            Dictionary with test results and metadata
        """
        print(f"=== {test_name.upper().replace('_', ' ')} SUITE ===")
        print(f"Strategy: {self.strategy.name}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        data = data_handler.get_data()
        print(
            f"Data loaded: {data.shape[0]} rows from {data['timestamp'][0]} to {data['timestamp'][-1]}"
        )
        print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

        self.strategy.set_data_handler(data_handler)

        signal_result = self.strategy.generate_signals()

        summary = self.strategy.get_strategy_summary(signal_result)
        print(f"Position Counts: {summary['position_counts']}")
        print(f"Signal Changes: {summary['signal_change_counts']}")
        print(f"Total Signals: {summary['total_signals']}")

        returns = self.strategy._calculate_strategy_returns(data, signal_result)

        results = {}
        for name, measure in self.measures.items():
            results[name] = measure.calculate(returns)

        print(f"\n=== PERFORMANCE RESULTS ===")
        for name, value in results.items():
            print(f"{name.title()}: {value:.4f}")

        test_metadata = {
            "test_name": test_name,
            "strategy_name": self.strategy.name,
            "test_date": datetime.now().isoformat(),
            "data_period": f"{data['timestamp'][0]} to {data['timestamp'][-1]}",
            "data_points": len(data),
            "strategy_summary": summary,
            "performance_results": results,
        }

        if run_version_id is None:
            run_version_id = self.version_manager.get_next_version("V")
        test_metadata["run_version"] = run_version_id
        self._run_dir = os.path.join(self.results_dir, run_version_id)
        os.makedirs(self._run_dir, exist_ok=True)

        self._save_results(test_metadata, run_version_id, test_name)

        print(f"\nRun output directory: {self._run_dir}")
        print(f"  {test_name}_performance_results.csv, {test_name}_metadata.json, ...")

        return test_metadata

    def create_performance_plots(
        self,
        data: pl.DataFrame,
        signal_result,
        results: Dict[str, Any],
        test_name: str = "insample_excellence",
        show_plot: bool = True,
        *,
        test_metadata: Dict[str, Any] | None = None,
    ):
        """Create performance visualizations using Bokeh (writes ``interactive.html`` into the run folder)."""
        print(f"\n=== CREATING PERFORMANCE PLOTS ===")

        custom_plots = []
        if hasattr(self.strategy, "create_custom_plots"):
            try:
                custom_plots = self.strategy.create_custom_plots(
                    data, signal_result, results=results
                )
                if custom_plots:
                    print(f"Found {len(custom_plots)} custom plot(s) from strategy")
            except Exception as e:
                print(f"Warning: Could not create custom plots: {e}")

        if self._run_dir is None:
            print(
                "\033[91mFAILED: No run directory (call run_test before create_performance_plots).\033[0m"
            )
            return

        try:
            bokeh_creator = BokehInteractivePlotCreator(self._run_dir)

            html_file = bokeh_creator.create_interactive_analysis(
                data,
                signal_result,
                results,
                test_name,
                show_plot,
                version_manager=None,
                custom_plots=custom_plots,
                strategy=self.strategy,
                html_filename=f"{test_name}_interactive.html",
            )

            if html_file is None:
                print(
                    "\033[91mFAILED: Interactive plot creation returned None. Check for data or plotting errors.\033[0m"
                )
            elif (
                test_metadata is not None
                and html_file
                and "versioned_files" in test_metadata
            ):
                test_metadata["versioned_files"]["plot_html"] = html_file

        except Exception as e:
            print(f"\033[91mFAILED: Error creating interactive plot: {e}\033[0m")
            print("Check your data and plotting configuration.")

    def _save_results(
        self, test_metadata: Dict[str, Any], run_version_id: str, test_name: str
    ):
        """Write CSV + JSON into ``self._run_dir`` (already created)."""
        assert self._run_dir is not None

        results_df = pl.DataFrame([test_metadata["performance_results"]])
        results_file = os.path.join(self._run_dir, f"{test_name}_performance_results.csv")
        results_df.write_csv(results_file)

        metadata_file = os.path.join(self._run_dir, f"{test_name}_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(test_metadata, f, indent=2, default=str)

        test_metadata["versioned_files"] = {
            "run_dir": self._run_dir,
            "run_version": run_version_id,
            "results_file": results_file,
            "metadata_file": metadata_file,
            "version": run_version_id,
        }

    def generate_test_report(
        self, test_metadata: Dict[str, Any], test_name: str = "insample_excellence"
    ):
        """Generate a markdown test report in the same run folder as metrics/plot."""
        if self._run_dir is None:
            raise RuntimeError("No run directory — call run_test before generate_test_report.")

        report_content = textwrap.dedent(f"""
            # {test_name.replace('_', ' ').title()} Suite Report

            **Test Date:** {test_metadata['test_date']}
            **Strategy:** {test_metadata['strategy_name']}
            **Data Period:** {test_metadata['data_period']}
            **Data Points:** {test_metadata['data_points']}

            ## Performance Results

            | Metric | Value |
            |--------|-------|
            | Total Return | {test_metadata['performance_results']['total_return']:.2%} |
            | Profit Factor | {test_metadata['performance_results']['profit_factor']:.2f} |
            | Sharpe Ratio | {test_metadata['performance_results']['sharpe_ratio']:.2f} |
            | Sortino Ratio | {test_metadata['performance_results']['sortino_ratio']:.2f} |
            | Max Drawdown | {test_metadata['performance_results']['max_drawdown']:.2%} |
            | Win Rate | {test_metadata['performance_results']['win_rate']:.2%} |

            ## Strategy Summary

            - **Total Signals:** {test_metadata['strategy_summary']['total_signals']}
            - **Position Counts:** {test_metadata['strategy_summary']['position_counts']}

            ## Signal Changes

            {test_metadata['strategy_summary']['signal_change_counts']}

            ## Analysis

            *Add your analysis and observations here...*

            ## Next Steps

            *Add next steps and recommendations here...*
        """).strip()

        report_file = os.path.join(self._run_dir, f"{test_name}_report.md")
        with open(report_file, "w") as f:
            f.write(report_content)

        if "versioned_files" in test_metadata:
            test_metadata["versioned_files"]["report_file"] = report_file

        metadata_file = os.path.join(self._run_dir, f"{test_name}_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(test_metadata, f, indent=2, default=str)

        print(f"Test report saved to: {report_file}")
        print(f"Updated metadata: {metadata_file}")
