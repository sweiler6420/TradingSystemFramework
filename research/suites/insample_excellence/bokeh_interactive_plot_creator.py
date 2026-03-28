"""
Bokeh Interactive Plot Creator
==============================

Creates truly interactive plots using Bokeh, which is specifically designed
for web-based interactive visualizations with excellent built-in saving.
"""

import html
import math
import os
import re
import webbrowser
from pathlib import Path

import numpy as np
import polars as pl
from typing import Dict, Any, Optional
from framework.signals import PositionState, SignalChange

# Bokeh imports
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CustomJS, Div, HoverTool, NumeralTickFormatter, Range1d, Spacer, Span
from bokeh.events import MouseMove
from bokeh.io import curdoc
from bokeh.resources import CDN
from bokeh.embed import file_html

# Dark HTML report (plot chrome + axes); switch to "light_minimal" for light exports
REPORT_BOKEH_THEME = "dark_minimal"

# ---------------------------------------------------------------------------
# Report visual overrides (easy to tune — softer than default dark_minimal)
# ---------------------------------------------------------------------------
# Page gutter (html/body): slightly darker than plot panels for depth / UX
REPORT_PAGE_BG = "#0a0f18"
# Cool blue-grey canvas — figures, summary card, Bokeh layout chrome (--background-color)
REPORT_PLOT_BG = "#0e1522"
REPORT_OUTER_BG = "#0e1522"
REPORT_GRID = "#1c2838"
# Text: mid grey — avoid near-white (#c9d1d9) on long reads
REPORT_AXIS_TEXT = "#7a8494"
REPORT_TITLE_TEXT = "#8b95a6"
# Muted series colors (avoid Category10 neons on dark bg)
COLOR_CLOSE_LINE = "#79b8ff"  # soft blue
COLOR_POSITION_LINE = "#b392f0"  # soft purple
COLOR_STRATEGY_EQ = "#7ce38b"  # soft green
COLOR_BUYHOLD_EQ = "#f9c66a"  # soft amber
COLOR_ENTRY = "#56d4a1"  # entry markers (muted mint)
COLOR_EXIT = "#f0a84a"  # exit markers (muted orange)
COLOR_LONG_AREA = "#2ea043"  # position long fill (GitHub-green-ish, used with alpha)
COLOR_SHORT_AREA = "#da3633"
COLOR_NEUTRAL_AREA = "#6e7681"
# Risk/reward rectangles (entry → exit): align with long/short area hues
COLOR_RR_PROFIT_ZONE = "#2ea043"
COLOR_RR_RISK_ZONE = "#da3633"
# Candlesticks — solid bodies + lighter, thicker wicks (read on dark REPORT_PLOT_BG)
COLOR_CANDLE_WICK = "#b4c2d4"
COLOR_CANDLE_WICK_WIDTH = 1.5
COLOR_CANDLE_UP_FILL = "#1e7f4f"
COLOR_CANDLE_UP_LINE = "#4ddb9a"
COLOR_CANDLE_DN_FILL = "#a83232"
COLOR_CANDLE_DN_LINE = "#ff8a80"
COLOR_METRIC_TEXT = "#7d8796"

# Standard OHLC LOD intervals available for candlestick rendering (label, minutes).
# Listed coarsest→finest; only intervals that are integer multiples of the data's
# base interval are actually built (e.g. 5m data → 1D / 4H / 1H / 30M / 15M / 5M).
_LOD_INTERVALS: list[tuple[str, int]] = [
    ("1D",  1440),
    ("4H",   240),
    ("1H",    60),
    ("30M",   30),
    ("15M",   15),
    ("5M",     5),
    ("1M",     1),
]
# Switch to the next finer interval when approximately this many bars become visible.
_LOD_TARGET_BARS = 800

# Kept as default for _resample_ohlcv_for_display's optional parameter.
MAX_CANDLE_DISPLAY_BARS = 5_000

# Main price + signals row is taller; other panes keep fixed heights below
REPORT_PRICE_SIGNAL_HEIGHT = 400
# Vertical gap between summary / each chart row (px)
REPORT_SECTION_GAP = 18
# Match performance summary card rounding (see _performance_summary_div)
REPORT_PLOT_CORNER_RADIUS = 10
# Y-axis default (position / equity panes): readable scalars.
REPORT_Y_AXIS_NUMERAL_FORMAT = "0,0.00"
# Price pane: tooltips + **price Y-axis** — enough decimals so grid labels are not all identical
# when levels differ by pips (e.g. avoid three ticks all showing "1.16").
REPORT_HOVER_PRICE_DECIMALS = 8
REPORT_HOVER_PRICE_NUMERAL = "0,0.00000000"
# Hover panel: match legend fill in _apply_report_figure_style (Bokeh tooltip reads CSS vars on body)
REPORT_HOVER_PANEL_BG = "#121b2c"

def _make_report_rounded_plot_stylesheet():
    """New ``InlineStyleSheet`` per Bokeh export — module-level singletons attach to one document."""
    try:
        from bokeh.models import InlineStyleSheet

        return InlineStyleSheet(
            css=f"""
:host {{
  border-radius: {REPORT_PLOT_CORNER_RADIUS}px;
  overflow: hidden;
  box-sizing: border-box;
}}
"""
        )
    except Exception:
        return None


def _make_performance_div_stylesheet():
    try:
        from bokeh.models import InlineStyleSheet

        return InlineStyleSheet(
            css="""
/* Bokeh MarkupView forces .bk-clearfix { display: inline-block } (inline style) — shrink-wraps HTML */
:host {
  display: block !important;
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  align-self: stretch !important;
  box-sizing: border-box;
}
:host .bk-clearfix {
  display: block !important;
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  box-sizing: border-box;
}
"""
        )
    except Exception:
        return None


# Injected into saved HTML so page background is set without relying on Jinja extends.
REPORT_PAGE_BG_MARKER = "<!-- tsf-report-page-bg -->"


def _inject_report_page_dark_style(html: str, page_bg: str, plot_chrome_bg: str) -> str:
    """
    After file_html(): inject page background (darker gutter) vs plot surface (figures unchanged).

    page_bg: html/body paint and inline attrs — margins around the report.
    plot_chrome_bg: Bokeh --background-color on roots (shadow hosts); matches figure panels.
    """
    style_block = f"""{REPORT_PAGE_BG_MARKER}
<style type="text/css">
html, body {{
  background-color: {page_bg} !important;
  margin: 0 !important;
  padding: 0 !important;
  min-height: 100%;
}}
body {{
  padding: 16px 20px 28px 20px !important;
  box-sizing: border-box;
  --background-color: {plot_chrome_bg} !important;
  /* HoverTool shadow host: align panel + default text with axes/legend (tuple tooltips use cyan labels) */
  --tooltip-text: {REPORT_AXIS_TEXT};
  --tooltip-color: {REPORT_HOVER_PANEL_BG};
  --tooltip-border: {REPORT_GRID};
}}
:root, html {{
  --background-color: {plot_chrome_bg} !important;
}}
body > div {{
  --background-color: {plot_chrome_bg} !important;
}}
</style>
"""

    if REPORT_PAGE_BG_MARKER not in html:
        m_head = re.search(r"(?i)</head>", html)
        if m_head:
            html = html[: m_head.start()] + "\n" + style_block + "\n" + html[m_head.start() :]

    m = re.search(r"(?i)</head>", html)
    if not m:
        return html

    before = html[: m.start()]
    close_head = m.group(0)
    after = html[m.end() :]

    attr = f'style="background-color:{page_bg} !important"'

    def _add_open_tag_attr(tag: str, chunk: str) -> str:
        def repl(match: re.Match[str]) -> str:
            inner = match.group(1)
            if re.search(r"\sstyle\s*=", inner, re.I):
                return match.group(0)
            return f"<{tag}{inner} {attr}>"

        return re.sub(rf"<{tag}([^>]*)>", repl, chunk, count=1, flags=re.I)

    before = _add_open_tag_attr("html", before)
    after = _add_open_tag_attr("body", after)
    return before + close_head + after


def _apply_report_figure_style(p) -> None:
    """Darker canvas + muted grid/axes/legend for long viewing sessions."""
    p.background_fill_color = REPORT_PLOT_BG
    p.border_fill_color = REPORT_OUTER_BG
    p.outline_line_color = None
    p.title.text_color = REPORT_TITLE_TEXT
    for ax in (p.xaxis, p.yaxis):
        ax.axis_label_text_color = REPORT_AXIS_TEXT
        ax.major_label_text_color = REPORT_AXIS_TEXT
        ax.major_tick_line_color = REPORT_GRID
        ax.minor_tick_line_color = None
    for g in (p.xgrid, p.ygrid):
        g.grid_line_color = REPORT_GRID
        g.grid_line_alpha = 0.35
        g.minor_grid_line_color = None
    leg = p.legend
    if leg:
        leg.background_fill_color = REPORT_HOVER_PANEL_BG
        leg.border_line_color = None
        leg.label_text_color = REPORT_AXIS_TEXT

    # Avoid scientific notation on linear y (e.g. BTC ~120000 → "120,000.00" not "1.2e+5").
    p.yaxis.formatter = NumeralTickFormatter(format=REPORT_Y_AXIS_NUMERAL_FORMAT)

    rounded_ss = _make_report_rounded_plot_stylesheet()
    if rounded_ss is not None and hasattr(p, "stylesheets"):
        existing = list(p.stylesheets) if p.stylesheets else []
        p.stylesheets = existing + [rounded_ss]


def _report_hover_tooltip_html(rows: list[tuple[str, str]]) -> str:
    """
    HTML HoverTool template with label/value colors matching axes and legend.

    Tuple-based tooltips use Bokeh's default table, which styles row labels cyan
    inside the tooltip shadow tree; HTML strings bypass that and match the report.
    """
    lab = REPORT_AXIS_TEXT
    val = REPORT_AXIS_TEXT
    out = [
        '<div style="display:table;border-spacing:4px 2px;font-size:13px;'
        "font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;\">"
    ]
    for label, field in rows:
        out.append(
            '<div style="display:table-row;">'
            f'<span style="display:table-cell;text-align:right;color:{lab};padding-right:8px;">'
            f"{html.escape(label)}</span>"
            f'<span style="display:table-cell;color:{val};">{field}</span>'
            "</div>"
        )
    out.append("</div>")
    return "".join(out)


def _fmt_hover_price_optional(x) -> str:
    """Format a price for tooltips; non-finite → em dash. Up to 8 dp, trim trailing zeros."""
    if x is None:
        return "—"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(xf) or math.isinf(xf):
        return "—"
    neg = xf < 0
    ax = abs(xf)
    raw = f"{ax:.{REPORT_HOVER_PRICE_DECIMALS}f}".rstrip("0").rstrip(".")
    if not raw:
        return "0"
    if "." in raw:
        ip, fp = raw.split(".", 1)
    else:
        ip, fp = raw, ""
    try:
        ip_fmt = f"{int(ip):,}"
    except ValueError:
        ip_fmt = ip
    out = ip_fmt + ("." + fp if fp else "")
    return ("-" if neg else "") + out


def _fmt_metric_value(x: float) -> str:
    """Format a scalar for the summary card; non-finite values won't break HTML."""
    if isinstance(x, (int, float)):
        if math.isinf(x):
            return "∞" if x > 0 else "-∞"
        if math.isnan(x):
            return "N/A"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "N/A"
    if math.isinf(xf):
        return "∞" if xf > 0 else "-∞"
    if math.isnan(xf):
        return "N/A"
    return f"{xf:.4f}"


def _equity_cumprod_stable(r) -> np.ndarray:
    """
    Cumulative product ∏(1+r) via log1p + cumsum + exp — avoids overflow on long series
    (plain (1+r).cum_prod() can hit inf and blank the Bokeh equity pane).
    """
    r = np.asarray(r, dtype=np.float64)
    r = np.where(np.isfinite(r), r, 0.0)
    r = np.clip(r, -0.999999, 50.0)
    return np.exp(np.cumsum(np.log1p(r)))


def _performance_summary_div(results: Dict[str, Any]) -> Div:
    """
    Plain HTML metrics block (not a figure) — label/value rows matching report styling.
    """
    def _to_float(key: str, default: float = 0.0) -> float:
        v = results.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    pf = _to_float("profit_factor")
    sharpe = _to_float("sharpe_ratio")
    sortino = _to_float("sortino_ratio")
    max_dd = _to_float("max_drawdown")
    total_return = _to_float("total_return")
    win_rate = _to_float("win_rate")

    bg = REPORT_PLOT_BG
    val_color = COLOR_METRIC_TEXT
    lab_color = REPORT_TITLE_TEXT

    def row(label: str, value: str) -> str:
        return (
            f'<div style="min-width:140px;">'
            f'<span style="color:{lab_color};font-size:11px;text-transform:uppercase;letter-spacing:0.05em;">'
            f"{html.escape(label)}</span><br/>"
            f'<span style="color:{val_color};font-size:15px;font-variant-numeric:tabular-nums;">'
            f"{html.escape(value)}</span>"
            f"</div>"
        )

    inner = (
        row("Profit factor", _fmt_metric_value(pf))
        + row("Sharpe", _fmt_metric_value(sharpe))
        + row("Sortino", _fmt_metric_value(sortino))
        + row("Max drawdown", _fmt_metric_value(max_dd))
        + row("Total return", _fmt_metric_value(total_return))
        + row("Win rate", _fmt_metric_value(win_rate))
    )

    block = f"""
<div style="display:flex;flex-direction:column;width:100%;max-width:100%;min-width:0;box-sizing:border-box;align-items:stretch;">
  <div style="width:100%;flex:1 1 auto;min-width:0;box-sizing:border-box;min-height:22vh;background-color:{bg};
  border-radius:{REPORT_PLOT_CORNER_RADIUS}px;padding:clamp(20px,3vh,36px) clamp(18px,4vw,40px);
  font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
    <div style="color:{lab_color};font-size:11px;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:14px;">
      Performance summary
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:18px 32px;align-items:flex-start;justify-content:flex-start;width:100%;">
      {inner}
    </div>
  </div>
</div>
"""
    ss = []
    perf_ss = _make_performance_div_stylesheet()
    if perf_ss is not None:
        ss.append(perf_ss)
    return Div(
        text=block.strip(),
        sizing_mode="stretch_width",
        width_policy="max",
        margin=0,
        stylesheets=ss,
    )


def _add_linked_vertical_crosshair(
    figures: list,
    *,
    line_color: str = REPORT_GRID,
    line_alpha: float = 0.9,
    line_width: int = 1,
) -> None:
    """
    Vertical line at pointer x, synchronized across stacked figures (same ``x_range``).

    Uses ``Span`` + ``MouseMove`` so one crosshair tracks horizontally across price,
    MACD, position, equity, etc.
    """
    if not figures:
        return
    spans: list[Span] = []
    for p in figures:
        span = Span(
            dimension="height",
            line_color=line_color,
            line_alpha=line_alpha,
            line_width=line_width,
            line_dash=[5, 3],
            visible=False,
        )
        p.add_layout(span)
        spans.append(span)

    args = {f"s{i}": s for i, s in enumerate(spans)}
    hide = "\n        ".join(f"s{i}.visible = false;" for i in range(len(spans)))
    show = "\n        ".join(
        f"s{i}.location = x;\n        s{i}.visible = true;" for i in range(len(spans))
    )
    code = f"""
    const x = cb_obj.x;
    if (x === null || x === undefined || (typeof x === 'number' && !isFinite(x))) {{
        {hide}
        return;
    }}
    {show}
    """
    cb = CustomJS(args=args, code=code)
    for p in figures:
        p.js_on_event(MouseMove, cb)


def _median_bar_width_ms(timestamps: pl.Series) -> float:
    """Bar width in milliseconds for Bokeh datetime axes (fraction of typical spacing)."""
    if len(timestamps) < 2:
        return 3600000.0
    t64 = timestamps.cast(pl.Datetime).to_numpy()
    diffs = np.diff(t64.astype("datetime64[ns]").astype(np.int64))
    med_ns = float(np.median(np.abs(diffs)))
    return med_ns * 0.8 / 1e6


def _position_side_scalar(x: float) -> int:
    """Map position signal to -1 / 0 / 1 (tolerates float enums)."""
    if not math.isfinite(x) or abs(x) < 0.25:
        return 0
    return 1 if x > 0 else -1


def _add_trade_risk_reward_zones(
    p,
    data: pl.DataFrame,
    signal_result,
    stop_s: pl.Series,
    tp_s: pl.Series,
    bar_width_ms: float,
) -> None:
    """
    Draw RR profit (green) and risk (red) quads from entry bar through exit bar for each
    open position, when ``get_trade_levels_for_plot`` supplies aligned stop/take-profit.

    Rendered behind price glyphs (caller should invoke before candles/lines).
    """
    n = len(data)
    if n == 0 or len(stop_s) != n or len(tp_s) != n:
        return

    pos = np.asarray(
        signal_result.position_signals.cast(pl.Float64).to_numpy(),
        dtype=np.float64,
    )
    if len(pos) != n:
        return

    close = np.asarray(data["close"].to_numpy(), dtype=np.float64)
    ts = data["timestamp"].to_numpy()
    slo = np.asarray(stop_s.to_numpy(), dtype=np.float64)
    tpo = np.asarray(tp_s.to_numpy(), dtype=np.float64)

    half_w = max(int(bar_width_ms * 0.5), 1)
    td_half = np.timedelta64(half_w, "ms")

    gl, gr, gb, gt = [], [], [], []
    rl, rr, rb, rt = [], [], [], []

    i = 0
    while i < n:
        side = _position_side_scalar(float(pos[i]))
        if side == 0:
            i += 1
            continue
        entry_i = i
        j = i + 1
        while j < n and _position_side_scalar(float(pos[j])) == side:
            j += 1
        exit_i = j - 1

        entry_px = float(close[entry_i])
        sl = float(slo[entry_i])
        tp = float(tpo[entry_i])
        if not all(math.isfinite(x) for x in (entry_px, sl, tp)):
            i = j
            continue

        if side == 1:
            if not (sl < entry_px < tp):
                i = j
                continue
            g_lo, g_hi = entry_px, tp
            r_lo, r_hi = sl, entry_px
        else:
            if not (tp < entry_px < sl):
                i = j
                continue
            g_lo, g_hi = tp, entry_px
            r_lo, r_hi = entry_px, sl

        t_left = ts[entry_i] - td_half
        t_right = ts[exit_i] + td_half

        gl.append(t_left)
        gr.append(t_right)
        gb.append(min(g_lo, g_hi))
        gt.append(max(g_lo, g_hi))
        rl.append(t_left)
        rr.append(t_right)
        rb.append(min(r_lo, r_hi))
        rt.append(max(r_lo, r_hi))

        i = j

    if gl:
        p.quad(
            left=gl,
            right=gr,
            bottom=gb,
            top=gt,
            fill_color=COLOR_RR_PROFIT_ZONE,
            fill_alpha=0.2,
            line_color=None,
            legend_label="RR profit zone",
        )
    if rl:
        p.quad(
            left=rl,
            right=rr,
            bottom=rb,
            top=rt,
            fill_color=COLOR_RR_RISK_ZONE,
            fill_alpha=0.2,
            line_color=None,
            legend_label="RR risk zone",
        )


def _resample_ohlcv_for_display(
    data: pl.DataFrame, max_bars: int = MAX_CANDLE_DISPLAY_BARS
) -> pl.DataFrame:
    """Aggregate OHLCV bars so the candle dataset never exceeds *max_bars* rows.

    Groups consecutive bars into equal-sized buckets and computes proper OHLC
    values (open=first, high=max, low=min, close=last) so bodies and wicks
    remain visually correct at the displayed resolution.  Returns *data*
    unchanged when it is already within the limit.
    """
    n = len(data)
    if n <= max_bars:
        return data
    step = math.ceil(n / max_bars)
    groups = np.arange(n, dtype=np.int64) // step
    return (
        data
        .with_columns(pl.Series("_display_grp", groups))
        .group_by("_display_grp")
        .agg([
            pl.col("timestamp").first(),
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
        ])
        .sort("timestamp")
        .drop("_display_grp")
    )


def _make_candle_cds(data: pl.DataFrame, bar_width_ms: float) -> ColumnDataSource:
    """Build a candle ColumnDataSource with pre-encoded up/down colors and bar_width.

    Storing colors as data columns (rather than using separate up/down renderers) lets
    a single wick segment + single vbar renderer swap resolution via CustomJS without
    needing separate renderers per direction.
    """
    open_np = np.asarray(data["open"].to_numpy(), dtype=np.float64)
    close_np = np.asarray(data["close"].to_numpy(), dtype=np.float64)
    inc = close_np >= open_np
    return ColumnDataSource({
        "timestamp": data["timestamp"].to_numpy(),
        "high": np.asarray(data["high"].to_numpy(), dtype=np.float64),
        "low": np.asarray(data["low"].to_numpy(), dtype=np.float64),
        "body_lo": np.minimum(open_np, close_np),
        "body_hi": np.maximum(open_np, close_np),
        "fill_color": np.where(inc, COLOR_CANDLE_UP_FILL, COLOR_CANDLE_DN_FILL),
        "line_color": np.where(inc, COLOR_CANDLE_UP_LINE, COLOR_CANDLE_DN_LINE),
        "bar_width": np.full(len(data), bar_width_ms),
    })


def _build_lod_cds_list(
    data: pl.DataFrame,
    w_ms: float,
) -> list[tuple[str, int, ColumnDataSource]]:
    """Build LOD (level-of-detail) candle datasets for standard time intervals.

    Returns ``[(label, threshold_ms, cds), ...]`` sorted **finest-first** so the
    caller can build a JS ``if / else if / else`` chain in order:

        if (span < threshold_finest)      → finest resolution
        else if (span < threshold_next)   → next coarser
        ...
        else                              → coarsest (fallback)

    ``threshold_ms`` is the visible x_range span (ms) at which this interval
    shows ~``_LOD_TARGET_BARS`` candles.  Only intervals that are integer
    multiples of the data's base interval are included, so the result length
    varies by dataset (e.g. 5m data → 5M/15M/30M/1H/4H/1D).
    """
    base_ms = w_ms / 0.8         # actual bar spacing (w_ms already applies 0.8 factor)
    base_min = base_ms / 60_000  # ms → minutes
    n = len(data)

    result: list[tuple[str, int, ColumnDataSource]] = []
    for label, interval_min in _LOD_INTERVALS:
        step_f = interval_min / base_min
        if step_f < 1.0 - 0.05:
            continue  # finer than base data — cannot aggregate down
        step = max(1, round(step_f))
        if step > 1 and abs(step - step_f) / step > 0.15:
            continue  # not a clean multiple of the base interval
        if step >= n:
            continue  # would collapse entire dataset to a single bar
        target_bars = max(1, n // step)
        res_df = _resample_ohlcv_for_display(data, target_bars) if step > 1 else data
        cds = _make_candle_cds(res_df, w_ms * step)
        # Threshold: this resolution fits ~_LOD_TARGET_BARS bars in the viewport
        threshold_ms = _LOD_TARGET_BARS * interval_min * 60_000
        result.append((label, threshold_ms, cds))

    # Sort finest-first (ascending threshold_ms) for the JS if/else chain
    result.sort(key=lambda x: x[1])
    return result


class BokehInteractivePlotCreator:
    """Creates interactive plots using Bokeh"""
    
    def __init__(self, plots_dir: str):
        self.plots_dir = plots_dir
        
    def create_interactive_analysis(
        self,
        data: pl.DataFrame,
        signal_result,
        results: Dict[str, Any],
        test_name: str = "analysis",
        show_plot: bool = True,
        version_manager=None,
        custom_plots=None,
        strategy=None,
        *,
        html_filename: str | None = None,
        symbol: str = "",
    ) -> Optional[str]:
        """Create interactive analysis using Bokeh.

        ``strategy`` may implement ``add_price_overlays(price_figure, data, signal_result, *, results=...)``
        to draw EMAs/bands on the main price pane (after OHLC and signals).

        If ``strategy.get_trade_levels_for_plot`` returns aligned stop/take-profit series,
        the price pane also draws green (profit) and red (risk) RR zones per trade behind
        the price series—no per-strategy chart code required.
        """
        
        try:
            print(f"\n=== CREATING BOKEH INTERACTIVE PLOT ===")
            
            # Prepare data
            data_copy = data.clone()
            # Add timestamp column if not present
            if 'timestamp' not in data_copy.columns:
                data_copy = data_copy.with_row_index('row_index')
                data_copy = data_copy.with_columns(
                    pl.col('row_index').cast(pl.Datetime).alias('timestamp')
                )
            
            # Versioned HTML lives next to other artifacts when ``version_manager`` is set.
            # Per-run folder: pass ``html_filename`` (e.g. ``{test_name}_interactive.html``).
            if version_manager:
                versioned_name = version_manager.get_versioned_filename(test_name, "html", "V")
                html_file = os.path.join(self.plots_dir, versioned_name)
            elif html_filename:
                html_file = os.path.join(self.plots_dir, html_filename)
            else:
                html_file = os.path.join(self.plots_dir, "interactive.html")
            
            # Theme applies to figures when serializing (dark axes, grid, outer area)
            curdoc().theme = REPORT_BOKEH_THEME

            # Strip common exchange prefixes (e.g. "X:BTCUSD" → "BTCUSD")
            _sym = re.sub(r'^[A-Za-z]:', '', symbol).strip() if symbol else ""

            # 1. Price and Signals Plot
            p1 = figure(
                title="",
                x_axis_type="datetime",
                height=REPORT_PRICE_SIGNAL_HEIGHT,
                sizing_mode="stretch_width",
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above",
                output_backend="webgl",
            )
            p1.title.text = _sym if _sym else "Price & Signals"
            p1.title.align = "center"
            p1.title.text_color = REPORT_AXIS_TEXT
            p1.title.text_font_size = "15px"
            p1.title.text_font_style = "normal"

            levels = None
            has_trade_levels = False
            stop_s = tp_s = None
            if strategy is not None:
                levels = strategy.get_trade_levels_for_plot(data_copy, signal_result)
            if levels is not None:
                stop_s, tp_s = levels
                if len(stop_s) == len(data_copy):
                    has_trade_levels = True

            w_ms = _median_bar_width_ms(data_copy["timestamp"])
            if has_trade_levels and stop_s is not None and tp_s is not None:
                _add_trade_risk_reward_zones(
                    p1, data_copy, signal_result, stop_s, tp_s, w_ms
                )

            # Candlesticks — standard-interval LOD (level-of-detail).
            #
            # One OHLC dataset per standard interval (1D / 4H / 1H / 30M / 15M / 5M / 1M)
            # is embedded in the HTML — only those that are integer multiples of the
            # data's base interval are built.  A CustomJS callback on x_range swaps the
            # active ColumnDataSource as the user zooms so the appropriate interval is
            # always displayed without a server.
            candle_renderers = []
            if all(c in data_copy.columns for c in ("open", "high", "low", "close")):
                lod_list = _build_lod_cds_list(data_copy, w_ms)
                # lod_list: [(label, threshold_ms, cds), ...] sorted finest-first

                if lod_list:
                    labels_built = [lbl for lbl, _, _ in lod_list]
                    print(f"Candle LOD intervals: {' / '.join(reversed(labels_built))}")

                    # Display CDS starts at coarsest for fast initial paint
                    initial_lod_label = lod_list[-1][0]
                    display_cds = ColumnDataSource(
                        data={k: v for k, v in lod_list[-1][2].data.items()}
                    )

                    # Update title to show ticker + initial (coarsest) interval
                    _title_sym = _sym if _sym else "Price & Signals"
                    p1.title.text = f"{_title_sym} | {initial_lod_label}"

                    wick_r = p1.segment(
                        x0="timestamp",
                        y0="high",
                        x1="timestamp",
                        y1="low",
                        source=display_cds,
                        line_color=COLOR_CANDLE_WICK,
                        line_width=COLOR_CANDLE_WICK_WIDTH,
                        line_alpha=1.0,
                        legend_label="Candlesticks",
                    )
                    body_r = p1.vbar(
                        x="timestamp",
                        width="bar_width",
                        bottom="body_lo",
                        top="body_hi",
                        fill_color="fill_color",
                        line_color="line_color",
                        fill_alpha=1.0,
                        line_alpha=1.0,
                        line_width=1,
                        source=display_cds,
                        legend_label="Candlesticks",
                    )
                    candle_renderers = [wick_r, body_r]
                    for r in candle_renderers:
                        r.visible = False

                    # Build JS args dict and if/else threshold chain dynamically.
                    # lod_list is finest-first so the if checks run finest → coarsest.
                    js_args: dict = {"display_cds": display_cds, "chart_title": p1.title}
                    js_ifs: list[str] = []
                    for i, (label, threshold_ms, cds) in enumerate(lod_list):
                        var = f"r{i}"
                        js_args[var] = cds
                        prefix = "if" if i == 0 else "else if"
                        js_ifs.append(
                            f"    {prefix} (span < {threshold_ms}) {{ src = r{i}; lod_label = '{label}'; }}"
                        )
                    js_ifs.append(
                        f"    else {{ src = r{len(lod_list) - 1}; lod_label = '{lod_list[-1][0]}'; }}"
                    )

                    # Embed symbol as a JS string literal (static, safe since it's our own value)
                    _js_sym_prefix = (_sym.replace("'", "\\'") + " | ") if _sym else ""
                    js_code = (
                        "const span = cb_obj.end - cb_obj.start;\n"
                        "if (!isFinite(span) || span <= 0) return;\n"
                        "let src; let lod_label;\n"
                        + "\n".join(js_ifs) + "\n"
                        + f"chart_title.text = '{_js_sym_prefix}' + lod_label;\n"
                        + "if (display_cds.data['timestamp'].length ==="
                        + " src.data['timestamp'].length) return;\n"
                        + "const nd = {};\n"
                        + "for (const k of Object.keys(src.data)) { nd[k] = src.data[k]; }\n"
                        + "display_cds.data = nd;\n"
                    )
                    _swap_js = CustomJS(args=js_args, code=js_code)
                    p1.x_range.js_on_change("start", _swap_js)
                    p1.x_range.js_on_change("end", _swap_js)

            # Close line: ColumnDataSource so hover can show RR stop / take profit when the strategy provides them
            close_cds_data: dict = {
                "timestamp": np.asarray(data_copy["timestamp"].to_numpy()),
                "close": np.asarray(data_copy["close"].to_numpy(), dtype=np.float64),
            }
            if has_trade_levels and stop_s is not None and tp_s is not None:
                sv = stop_s.to_numpy()
                tv = tp_s.to_numpy()
                close_cds_data["stop_loss_str"] = [_fmt_hover_price_optional(x) for x in sv]
                close_cds_data["take_profit_str"] = [_fmt_hover_price_optional(x) for x in tv]
            cds_close = ColumnDataSource(data=close_cds_data)
            close_r = p1.line(
                x="timestamp",
                y="close",
                source=cds_close,
                line_width=2,
                color=COLOR_CLOSE_LINE,
                legend_label="Close (line)",
            )

            entry_r = None
            exit_r = None
            if has_trade_levels and stop_s is not None and tp_s is not None:
                plot_data = signal_result.get_signal_changes_for_plotting(
                    data_copy, stop_loss=stop_s, take_profit=tp_s
                )
            else:
                plot_data = signal_result.get_signal_changes_for_plotting(data_copy)
            if len(plot_data) > 0:
                bar_indices = plot_data["index"].to_list()
                timestamps = plot_data["timestamp"].to_list()
                prices = plot_data["price"].to_list()
                signal_changes = plot_data["signal_change"].to_list()

                entry_timestamps = []
                entry_prices = []
                entry_labels: list[str] = []
                entry_sl_str: list[str] = []
                entry_tp_str: list[str] = []
                exit_timestamps = []
                exit_prices = []

                sv_arr = tv_arr = None
                if has_trade_levels and stop_s is not None and tp_s is not None:
                    sv_arr = stop_s.to_numpy()
                    tv_arr = tp_s.to_numpy()

                for j, signal in enumerate(signal_changes):
                    signal_str = str(signal)
                    bar_idx = int(bar_indices[j])
                    if "TO_LONG" in signal_str or "TO_SHORT" in signal_str:
                        entry_timestamps.append(timestamps[j])
                        entry_prices.append(prices[j])
                        entry_labels.append(
                            "Long entry" if "TO_LONG" in signal_str else "Short entry"
                        )
                        if sv_arr is not None and tv_arr is not None and 0 <= bar_idx < len(
                            sv_arr
                        ):
                            entry_sl_str.append(
                                _fmt_hover_price_optional(sv_arr[bar_idx])
                            )
                            entry_tp_str.append(
                                _fmt_hover_price_optional(tv_arr[bar_idx])
                            )
                        else:
                            entry_sl_str.append("—")
                            entry_tp_str.append("—")
                    elif "TO_NEUTRAL" in signal_str:
                        exit_timestamps.append(timestamps[j])
                        exit_prices.append(prices[j])

                if entry_timestamps:
                    entry_dict = dict(
                        timestamp=entry_timestamps,
                        price=entry_prices,
                        signal_label=entry_labels,
                        sl_str=entry_sl_str,
                        tp_str=entry_tp_str,
                    )
                    entry_src = ColumnDataSource(data=entry_dict)
                    entry_r = p1.scatter(
                        x="timestamp",
                        y="price",
                        source=entry_src,
                        size=14,
                        color=COLOR_ENTRY,
                        marker="triangle",
                        legend_label="Entry Signals",
                        alpha=0.85,
                        line_color="#1a3328",
                        line_width=0.5,
                    )

                if exit_timestamps:
                    exit_src = ColumnDataSource(
                        data=dict(
                            timestamp=exit_timestamps,
                            price=exit_prices,
                            signal_label=["Exit"] * len(exit_timestamps),
                        )
                    )
                    exit_r = p1.scatter(
                        x="timestamp",
                        y="price",
                        source=exit_src,
                        size=14,
                        color=COLOR_EXIT,
                        marker="inverted_triangle",
                        legend_label="Exit Signals",
                        alpha=0.85,
                        line_color="#4a3318",
                        line_width=0.5,
                    )

            _apply_report_figure_style(p1)
            # Override default 2-decimal Y formatter so price ticks match instrument precision.
            p1.yaxis.formatter = NumeralTickFormatter(format=REPORT_HOVER_PRICE_NUMERAL)
            p1.legend.location = "top_left"
            p1.legend.click_policy = "hide"
            p1.yaxis.axis_label = "Price ($)"

            if strategy is not None:
                try:
                    strategy.add_price_overlays(
                        p1,
                        data_copy,
                        signal_result,
                        results=results,
                    )
                except Exception as e:
                    print(f"Warning: add_price_overlays failed: {e}")
            
            # 2. Position States Plot
            p2 = figure(
                title="Position States",
                x_axis_type="datetime",
                height=200,
                sizing_mode="stretch_width",
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above",
                x_range=p1.x_range,  # Link x-axis
                output_backend="webgl",
            )
            
            # Position areas (optimized, no warnings)
            # Convert to numeric values once for efficiency
            numeric_positions = signal_result.position_signals.cast(pl.Float64)
            
            p2.line(
                data_copy["timestamp"],
                numeric_positions,
                line_width=2,
                color=COLOR_POSITION_LINE,
                legend_label="Position",
            )

            # Add area fills (optimized)
            long_mask = numeric_positions == 1
            short_mask = numeric_positions == -1

            p2.varea(
                data_copy["timestamp"],
                0.5,
                np.where(long_mask, 1, np.nan),
                color=COLOR_LONG_AREA,
                alpha=0.22,
                legend_label="Long",
            )

            p2.varea(
                data_copy["timestamp"],
                -0.5,
                np.where(short_mask, -1, np.nan),
                color=COLOR_SHORT_AREA,
                alpha=0.22,
                legend_label="Short",
            )

            p2.varea(
                data_copy["timestamp"],
                -0.5,
                0.5,
                color=COLOR_NEUTRAL_AREA,
                alpha=0.18,
                legend_label="Neutral",
            )

            _apply_report_figure_style(p2)
            p2.legend.location = "top_left"
            p2.legend.click_policy = "hide"
            p2.yaxis.axis_label = "Position"
            p2.y_range = Range1d(-1.5, 1.5)
            
            # 3. Equity Curve Plot
            p3 = figure(
                title="Equity Curve Comparison",
                x_axis_type="datetime",
                height=200,
                sizing_mode="stretch_width",
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above",
                x_range=p1.x_range,  # Link x-axis
                output_backend="webgl",
            )
            
            # Equity: match framework / strategy returns (includes RR exit fills when implemented).
            numeric_signals = signal_result.position_signals.cast(pl.Float64)
            if strategy is not None and hasattr(strategy, "_calculate_strategy_returns"):
                returns = strategy._calculate_strategy_returns(data_copy, signal_result)
            else:
                returns = numeric_signals * np.log(data_copy["close"]).diff().shift(-1)
            r_strat = np.asarray(returns.to_numpy(), dtype=np.float64)
            r_bh = np.asarray(
                (np.log(data_copy["close"]).diff().shift(-1)).to_numpy(),
                dtype=np.float64,
            )
            cumulative_returns = _equity_cumprod_stable(np.nan_to_num(r_strat, nan=0.0))
            buy_hold_returns = _equity_cumprod_stable(np.nan_to_num(r_bh, nan=0.0))

            p3.line(
                data_copy["timestamp"],
                cumulative_returns,
                line_width=2,
                color=COLOR_STRATEGY_EQ,
                legend_label="Strategy",
            )
            p3.line(
                data_copy["timestamp"],
                buy_hold_returns,
                line_width=2,
                color=COLOR_BUYHOLD_EQ,
                legend_label="Buy & Hold",
            )

            _apply_report_figure_style(p3)
            p3.legend.location = "top_left"
            p3.legend.click_policy = "hide"
            p3.yaxis.axis_label = "Cumulative Returns"

            # 4. Performance summary — HTML text block (not a figure)
            performance_summary = _performance_summary_div(results)

            # Price pane: separate hovers so close line can show stop/TP columns without breaking markers
            if has_trade_levels:
                price_tip_rows = [
                    ("Date", "@timestamp{%F}"),
                    ("Price", f"@close{{{REPORT_HOVER_PRICE_NUMERAL}}}"),
                    ("Stop loss", "@stop_loss_str"),
                    ("Take profit", "@take_profit_str"),
                ]
            else:
                price_tip_rows = [
                    ("Date", "@timestamp{%F}"),
                    ("Price", f"@close{{{REPORT_HOVER_PRICE_NUMERAL}}}"),
                ]
            hover_close = HoverTool(
                renderers=[close_r],
                tooltips=_report_hover_tooltip_html(price_tip_rows),
            )
            hover_close.formatters = {"@timestamp": "datetime"}
            p1.add_tools(hover_close)
            # Entry markers: show intended SL / TP at open (values keyed by bar index from plot_data)
            if entry_r is not None:
                entry_tip_rows = [
                    ("Date", "@timestamp{%F}"),
                    ("Price", f"@price{{{REPORT_HOVER_PRICE_NUMERAL}}}"),
                    ("Signal", "@signal_label"),
                    ("SL", "@sl_str"),
                    ("TP", "@tp_str"),
                ]
                hover_entry = HoverTool(
                    renderers=[entry_r],
                    tooltips=_report_hover_tooltip_html(entry_tip_rows),
                )
                hover_entry.formatters = {"@timestamp": "datetime"}
                p1.add_tools(hover_entry)
            if exit_r is not None:
                hover_exit = HoverTool(
                    renderers=[exit_r],
                    tooltips=_report_hover_tooltip_html(
                        [
                            ("Date", "@timestamp{%F}"),
                            ("Price", f"@price{{{REPORT_HOVER_PRICE_NUMERAL}}}"),
                            ("Signal", "@signal_label"),
                        ]
                    ),
                )
                hover_exit.formatters = {"@timestamp": "datetime"}
                p1.add_tools(hover_exit)

            hover2 = HoverTool(
                tooltips=_report_hover_tooltip_html([("Date", "@x{%F}"), ("Position", "@y")])
            )
            hover2.formatters = {"@x": "datetime"}
            p2.add_tools(hover2)

            hover3 = HoverTool(
                tooltips=_report_hover_tooltip_html(
                    [("Date", "@x{%F}"), ("Returns", "@y{0.0000}")]
                )
            )
            hover3.formatters = {"@x": "datetime"}
            p3.add_tools(hover3)
            
            # Stack: performance summary (top) → price+signals → feature panes → position → equity
            plot_stack = [p1]
            if custom_plots:
                for custom_plot in custom_plots:
                    custom_plot.x_range = p1.x_range
                    _apply_report_figure_style(custom_plot)
                plot_stack.extend(custom_plots)
            plot_stack.extend([p2, p3])

            _add_linked_vertical_crosshair(plot_stack)

            layout_children = [Spacer(height=8)]
            for i, child in enumerate([performance_summary] + plot_stack):
                if i > 0:
                    layout_children.append(Spacer(height=REPORT_SECTION_GAP))
                layout_children.append(child)
            layout_children.append(Spacer(height=12))

            layout = column(*layout_children, sizing_mode="stretch_width")
            
            _report_title = f"{test_name.replace('_', ' ').title()} - Interactive Analysis"
            html_out = file_html(layout, CDN, title=_report_title)
            html_out = _inject_report_page_dark_style(html_out, REPORT_PAGE_BG, REPORT_PLOT_BG)
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html_out)

            # Open the saved HTML (same bytes as on disk). show(layout) would open a new
            # Bokeh document without our file_html post-processing (no page background).
            if show_plot:
                try:
                    webbrowser.open(Path(html_file).resolve().as_uri())
                    print("Opened saved interactive report in browser.")
                except Exception as e:
                    print(f"Could not open report in browser: {e}")
            
            print(f"Bokeh interactive plot saved to: {html_file}")
            print("Open this file in a web browser for full interactivity!")
            
            return html_file
            
        except ImportError:
            print("Bokeh not available. Install with: pip install bokeh")
            return None
        except Exception as e:
            print(f"Error creating Bokeh plot: {e}")
            return None
    
