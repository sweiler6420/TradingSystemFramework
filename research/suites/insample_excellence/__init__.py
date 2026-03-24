"""In-sample excellence suite: metrics, Bokeh reports, versioned artifacts."""

from .insample_excellence_suite import InSampleExcellenceSuite

# Backward-compatible alias
InSampleExcellenceTest = InSampleExcellenceSuite

__all__ = ["InSampleExcellenceSuite", "InSampleExcellenceTest"]
