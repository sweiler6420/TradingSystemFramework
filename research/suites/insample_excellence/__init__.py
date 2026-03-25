"""In-sample excellence suite: metrics, Bokeh reports, versioned artifacts."""

from .batch_equity_plot import (
    BATCH_EQUITY_HTML_NAME,
    build_batch_equity_figure,
    join_equity_frames,
    write_batch_equity_html,
)
from .batch_summary_report import (
    compute_equal_weight_portfolio_metrics,
    load_insample_metadata_from_run_dir,
    regenerate_batch_summary,
    render_batch_html,
    write_insample_batch_summary,
)
from .insample_excellence_suite import InSampleExcellenceSuite

# Backward-compatible alias
InSampleExcellenceTest = InSampleExcellenceSuite

__all__ = [
    "InSampleExcellenceSuite",
    "InSampleExcellenceTest",
    "BATCH_EQUITY_HTML_NAME",
    "build_batch_equity_figure",
    "join_equity_frames",
    "write_batch_equity_html",
    "compute_equal_weight_portfolio_metrics",
    "load_insample_metadata_from_run_dir",
    "regenerate_batch_summary",
    "render_batch_html",
    "write_insample_batch_summary",
]
