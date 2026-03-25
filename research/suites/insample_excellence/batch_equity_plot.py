"""
Multi-underlying equity comparison (Bokeh): per-name buy & hold + strategy, plus equal-weight
blends of strategy and of buy & hold (when ≥2 names) for combined-vs-combined comparison.

Legend uses Bokeh ``click_policy=\"hide\"`` so any series can be toggled off.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from bokeh.embed import file_html
from bokeh.models import NumeralTickFormatter
from bokeh.plotting import curdoc
from bokeh.resources import CDN

from .bokeh_interactive_plot_creator import (
    REPORT_BOKEH_THEME,
    REPORT_PAGE_BG,
    REPORT_PLOT_BG,
    _inject_report_page_dark_style,
    _apply_report_figure_style,
)

BATCH_EQUITY_HTML_NAME = "insample_excellence_batch_equity.html"

# Distinct hues per underlying; buy & hold uses the same hue blended toward the plot surface (muted).
_LINE_COLORS = [
    "#79b8ff",
    "#b392f0",
    "#f9c66a",
    "#56d4a1",
    "#f0a84a",
    "#b4c2d4",
    "#ff8a80",
    "#4ddb9a",
]
_BLEND_STRATEGY_COLOR = "#7ce38b"
_BLEND_BUYHOLD_COLOR = "#f9c66a"

# Pull vivid line colors toward panel bg so buy & hold reads softer without dash/dot patterns.
_MUTE_TOWARD_BG = 0.4


def _hex_rgb(h: str) -> tuple[int, int, int]:
    s = h.strip().lstrip("#")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _rgb_hex(r: int, g: int, b: int) -> str:
    return f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}"


def _muted_line_color(hex_color: str, *, toward: str = REPORT_PLOT_BG, amount: float = _MUTE_TOWARD_BG) -> str:
    """Blend ``hex_color`` toward ``toward`` (linear RGB) for a softer same-hue read on dark charts."""
    r1, g1, b1 = _hex_rgb(hex_color)
    r2, g2, b2 = _hex_rgb(toward)
    t = max(0.0, min(1.0, amount))
    return _rgb_hex(
        int(round(r1 + (r2 - r1) * t)),
        int(round(g1 + (g2 - g1) * t)),
        int(round(b1 + (b2 - b1) * t)),
    )


def _equity_cumprod_stable(r: np.ndarray) -> np.ndarray:
    """Same as interactive report: ∏(1+r) via log1p stabilized."""
    r = np.asarray(r, dtype=np.float64)
    r = np.where(np.isfinite(r), r, 0.0)
    r = np.clip(r, -0.999999, 50.0)
    return np.exp(np.cumsum(np.log1p(r)))


def join_equity_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame()
    acc = frames[0]
    for f in frames[1:]:
        acc = acc.join(f, on="timestamp", how="inner")
    return acc


def build_batch_equity_figure(
    wide: pl.DataFrame,
    slug_to_label: dict[str, str],
    *,
    title: str = "Batch equity comparison",
):
    """
    Build the multi-underlying equity ``figure``, or ``None`` if ``wide`` cannot be plotted.

    ``wide`` must contain ``timestamp`` and, for each slug ``k``, columns ``strat_{k}`` and ``bh_{k}``.
    When ≥2 names align, adds equal-weight blends of strategy returns and of buy-and-hold returns
    (same mean-of-returns rule as the batch summary blended path).
    """
    from bokeh.plotting import figure as bk_figure

    if wide.is_empty() or "timestamp" not in wide.columns:
        return None

    strat_cols = sorted(c for c in wide.columns if c.startswith("strat_"))
    if not strat_cols:
        return None

    curdoc().theme = REPORT_BOKEH_THEME

    ts = wide["timestamp"]
    ts_py = ts.to_list()

    p = bk_figure(
        title=title,
        x_axis_type="datetime",
        height=440,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        toolbar_location="above",
    )

    for i, sc in enumerate(strat_cols):
        slug = sc[len("strat_") :]
        bc = f"bh_{slug}"
        if bc not in wide.columns:
            continue
        label_base = slug_to_label.get(slug, slug)
        color = _LINE_COLORS[i % len(_LINE_COLORS)]
        color_bh = _muted_line_color(color)
        r_s = np.asarray(wide[sc].to_numpy(), dtype=np.float64)
        r_bh = np.asarray(wide[bc].to_numpy(), dtype=np.float64)
        eq_s = _equity_cumprod_stable(np.nan_to_num(r_s, nan=0.0))
        eq_bh = _equity_cumprod_stable(np.nan_to_num(r_bh, nan=0.0))

        p.line(
            ts_py,
            eq_bh,
            line_width=2,
            color=color_bh,
            alpha=0.95,
            legend_label=f"{label_base} · buy & hold",
        )
        p.line(
            ts_py,
            eq_s,
            line_width=2,
            color=color,
            alpha=0.95,
            legend_label=f"{label_base} · strategy",
        )

    if len(strat_cols) >= 2:
        pr = wide.select(
            (pl.sum_horizontal([pl.col(c) for c in strat_cols]) / float(len(strat_cols))).alias(
                "_blend_r"
            )
        )["_blend_r"].to_numpy()
        pr = np.asarray(pr, dtype=np.float64)
        eq_blend = _equity_cumprod_stable(np.nan_to_num(pr, nan=0.0))
        p.line(
            ts_py,
            eq_blend,
            line_width=3,
            color=_BLEND_STRATEGY_COLOR,
            alpha=0.95,
            legend_label="Equal-weight blend (strategy)",
        )

        bh_cols: list[str] = []
        for sc in strat_cols:
            slug = sc[len("strat_") :]
            bc = f"bh_{slug}"
            if bc in wide.columns:
                bh_cols.append(bc)
        if len(bh_cols) >= 2:
            pbh = wide.select(
                (pl.sum_horizontal([pl.col(c) for c in bh_cols]) / float(len(bh_cols))).alias(
                    "_blend_bh_r"
                )
            )["_blend_bh_r"].to_numpy()
            pbh = np.asarray(pbh, dtype=np.float64)
            eq_blend_bh = _equity_cumprod_stable(np.nan_to_num(pbh, nan=0.0))
            p.line(
                ts_py,
                eq_blend_bh,
                line_width=3,
                color=_muted_line_color(_BLEND_BUYHOLD_COLOR),
                alpha=0.95,
                legend_label="Equal-weight blend (buy & hold)",
            )

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.yaxis.axis_label = "Cumulative growth (start = 1)"
    p.xaxis.axis_label = "Time"
    _apply_report_figure_style(p)
    return p


def write_batch_equity_html(
    out_path: str | Path,
    wide: pl.DataFrame,
    slug_to_label: dict[str, str],
    *,
    title: str = "Batch equity comparison",
) -> Path:
    """Write a standalone HTML file (full document). Prefer embedding via batch summary when possible."""
    path = Path(out_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = build_batch_equity_figure(wide, slug_to_label, title=title)
    if fig is None:
        if wide.is_empty() or "timestamp" not in wide.columns:
            msg = "No aligned timestamps for equity plot."
        elif not sorted(c for c in wide.columns if c.startswith("strat_")):
            msg = "No strategy return columns."
        else:
            msg = "No equity plot data."
        path.write_text(
            f"<!DOCTYPE html><html><body><p>{msg}</p></body></html>",
            encoding="utf-8",
        )
        return path

    html_out = file_html(fig, CDN, title=title)
    html_out = _inject_report_page_dark_style(html_out, REPORT_PAGE_BG, REPORT_PLOT_BG)
    path.write_text(html_out, encoding="utf-8")
    return path
