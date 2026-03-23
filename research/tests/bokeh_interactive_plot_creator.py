"""
Bokeh Interactive Plot Creator
==============================

Creates truly interactive plots using Bokeh, which is specifically designed
for web-based interactive visualizations with excellent built-in saving.
"""

import html
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
from bokeh.models import Div, HoverTool, Range1d, Spacer
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
# Candlesticks (desaturated)
COLOR_CANDLE_WICK = "#6e7681"
COLOR_CANDLE_UP_FILL = "#3d5a4a"
COLOR_CANDLE_UP_LINE = "#5a7d6a"
COLOR_CANDLE_DN_FILL = "#6a3d3d"
COLOR_CANDLE_DN_LINE = "#8a5a5a"
COLOR_METRIC_TEXT = "#7d8796"

# Main price + signals row is taller; other panes keep fixed heights below
REPORT_PRICE_SIGNAL_HEIGHT = 400
# Vertical gap between summary / each chart row (px)
REPORT_SECTION_GAP = 18
# Match performance summary card rounding (see _performance_summary_div)
REPORT_PLOT_CORNER_RADIUS = 10

try:
    from bokeh.models import InlineStyleSheet

    _REPORT_ROUNDED_PLOT_STYLESHEET = InlineStyleSheet(
        css=f"""
:host {{
  border-radius: {REPORT_PLOT_CORNER_RADIUS}px;
  overflow: hidden;
  box-sizing: border-box;
}}
"""
    )
    _PERFORMANCE_DIV_STYLESHEET = InlineStyleSheet(
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
    _REPORT_ROUNDED_PLOT_STYLESHEET = None
    _PERFORMANCE_DIV_STYLESHEET = None


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
        leg.background_fill_color = "#121b2c"
        leg.border_line_color = None
        leg.label_text_color = REPORT_AXIS_TEXT

    if _REPORT_ROUNDED_PLOT_STYLESHEET is not None and hasattr(p, "stylesheets"):
        existing = list(p.stylesheets) if p.stylesheets else []
        if _REPORT_ROUNDED_PLOT_STYLESHEET not in existing:
            p.stylesheets = existing + [_REPORT_ROUNDED_PLOT_STYLESHEET]


def _performance_summary_div(results: Dict[str, Any]) -> Div:
    """
    Plain HTML metrics block (not a figure) — label/value rows matching report styling.
    """
    pf = float(results.get("profit_factor") or 0)
    sharpe = float(results.get("sharpe_ratio") or 0)
    sortino = float(results.get("sortino_ratio") or 0)
    max_dd = float(results.get("max_drawdown") or 0)
    total_return = float(results.get("total_return") or 0)
    win_rate = float(results.get("win_rate") or 0)

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
        row("Profit factor", f"{pf:.4f}")
        + row("Sharpe", f"{sharpe:.4f}")
        + row("Sortino", f"{sortino:.4f}")
        + row("Max drawdown", f"{max_dd:.4f}")
        + row("Total return", f"{total_return:.4f}")
        + row("Win rate", f"{win_rate:.4f}")
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
    if _PERFORMANCE_DIV_STYLESHEET is not None:
        ss.append(_PERFORMANCE_DIV_STYLESHEET)
    return Div(
        text=block.strip(),
        sizing_mode="stretch_width",
        width_policy="max",
        margin=0,
        stylesheets=ss,
    )


def _median_bar_width_ms(timestamps: pl.Series) -> float:
    """Bar width in milliseconds for Bokeh datetime axes (fraction of typical spacing)."""
    if len(timestamps) < 2:
        return 3600000.0
    t64 = timestamps.cast(pl.Datetime).to_numpy()
    diffs = np.diff(t64.astype("datetime64[ns]").astype(np.int64))
    med_ns = float(np.median(np.abs(diffs)))
    return med_ns * 0.8 / 1e6


class BokehInteractivePlotCreator:
    """Creates interactive plots using Bokeh"""
    
    def __init__(self, plots_dir: str):
        self.plots_dir = plots_dir
        
    def create_interactive_analysis(self, data: pl.DataFrame, signal_result, 
                                   results: Dict[str, Any], test_name: str = "analysis", 
                                   show_plot: bool = True, version_manager=None, custom_plots=None) -> Optional[str]:
        """Create interactive analysis using Bokeh"""
        
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
            
            # Create output file with versioning
            if version_manager:
                versioned_name = version_manager.get_versioned_filename(test_name, "html", "V")
                html_file = f"{self.plots_dir}/{versioned_name}"
            else:
                html_file = f"{self.plots_dir}/{test_name}_bokeh_interactive.html"
            
            # Theme applies to figures when serializing (dark axes, grid, outer area)
            curdoc().theme = REPORT_BOKEH_THEME

            # 1. Price and Signals Plot
            p1 = figure(
                title="Price and Signal Changes",
                x_axis_type="datetime",
                height=REPORT_PRICE_SIGNAL_HEIGHT,
                sizing_mode="stretch_width",
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above",
            )
            
            # Candlesticks (OHLC): one legend group, hidden by default; toggle on when zoomed
            if all(c in data_copy.columns for c in ("open", "high", "low", "close")):
                ts_np = data_copy["timestamp"].to_numpy()
                # Polars <1.0: to_numpy() has no dtype= kwarg; cast via numpy
                open_np = np.asarray(data_copy["open"].to_numpy(), dtype=np.float64)
                high_np = np.asarray(data_copy["high"].to_numpy(), dtype=np.float64)
                low_np = np.asarray(data_copy["low"].to_numpy(), dtype=np.float64)
                close_np = np.asarray(data_copy["close"].to_numpy(), dtype=np.float64)
                w_ms = _median_bar_width_ms(data_copy["timestamp"])
                body_lo = np.minimum(open_np, close_np)
                body_hi = np.maximum(open_np, close_np)
                inc = close_np >= open_np

                # Use legend_label (not legend_group): legend_group expects a *column name*
                # in a ColumnDataSource, not a display string.
                wick_r = p1.segment(
                    x0=ts_np,
                    y0=high_np,
                    x1=ts_np,
                    y1=low_np,
                    line_color=COLOR_CANDLE_WICK,
                    line_width=1,
                    legend_label="Candlesticks",
                )
                candle_renderers = [wick_r]
                if np.any(inc):
                    up_r = p1.vbar(
                        x=ts_np[inc],
                        width=w_ms,
                        bottom=body_lo[inc],
                        top=body_hi[inc],
                        fill_color=COLOR_CANDLE_UP_FILL,
                        line_color=COLOR_CANDLE_UP_LINE,
                        legend_label="Candlesticks",
                    )
                    candle_renderers.append(up_r)
                if np.any(~inc):
                    down_r = p1.vbar(
                        x=ts_np[~inc],
                        width=w_ms,
                        bottom=body_lo[~inc],
                        top=body_hi[~inc],
                        fill_color=COLOR_CANDLE_DN_FILL,
                        line_color=COLOR_CANDLE_DN_LINE,
                        legend_label="Candlesticks",
                    )
                    candle_renderers.append(down_r)
                for r in candle_renderers:
                    r.visible = False

            # Close line (toggle off when using candlesticks zoomed in)
            p1.line(
                data_copy["timestamp"],
                data_copy["close"],
                line_width=2,
                color=COLOR_CLOSE_LINE,
                legend_label="Close (line)",
            )
            
            # Signal markers (optimized with ColumnDataSource)
            plot_data = signal_result.get_signal_changes_for_plotting(data_copy)
            if len(plot_data) > 0:
                # Convert to lists for filtering
                timestamps = plot_data['timestamp'].to_list()
                prices = plot_data['price'].to_list()
                signal_changes = plot_data['signal_change'].to_list()
                
                # Separate signals by type
                entry_timestamps = []
                entry_prices = []
                exit_timestamps = []
                exit_prices = []
                
                for i, signal in enumerate(signal_changes):
                    signal_str = str(signal)
                    if 'TO_LONG' in signal_str or 'TO_SHORT' in signal_str:
                        entry_timestamps.append(timestamps[i])
                        entry_prices.append(prices[i])
                    elif 'TO_NEUTRAL' in signal_str:
                        exit_timestamps.append(timestamps[i])
                        exit_prices.append(prices[i])
                
                # Plot entry signals (up arrows)
                if entry_timestamps:
                    p1.scatter(
                        entry_timestamps,
                        entry_prices,
                        size=14,
                        color=COLOR_ENTRY,
                        marker="triangle",
                        legend_label="Entry Signals",
                        alpha=0.85,
                        line_color="#1a3328",
                        line_width=0.5,
                    )

                if exit_timestamps:
                    p1.scatter(
                        exit_timestamps,
                        exit_prices,
                        size=14,
                        color=COLOR_EXIT,
                        marker="inverted_triangle",
                        legend_label="Exit Signals",
                        alpha=0.85,
                        line_color="#4a3318",
                        line_width=0.5,
                    )

            _apply_report_figure_style(p1)
            p1.legend.location = "top_left"
            p1.legend.click_policy = "hide"
            p1.yaxis.axis_label = "Price ($)"
            
            # 2. Position States Plot
            p2 = figure(
                title="Position States",
                x_axis_type="datetime",
                height=200,
                sizing_mode="stretch_width",
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above",
                x_range=p1.x_range,  # Link x-axis
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
            )
            
            # Calculate returns (optimized conversion, no warnings)
            numeric_signals = signal_result.position_signals.cast(pl.Float64)
            returns = numeric_signals * np.log(data_copy['close']).diff().shift(-1)
            cumulative_returns = (1 + returns).cum_prod()
            buy_hold_returns = (1 + np.log(data_copy['close']).diff().shift(-1)).cum_prod()
            
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

            # Add hover tools
            hover1 = HoverTool(tooltips=[("Date", "@x{%F}"), ("Price", "@y{$0,0.00}")])
            hover1.formatters = {"@x": "datetime"}
            p1.add_tools(hover1)
            
            hover2 = HoverTool(tooltips=[("Date", "@x{%F}"), ("Position", "@y")])
            hover2.formatters = {"@x": "datetime"}
            p2.add_tools(hover2)
            
            hover3 = HoverTool(tooltips=[("Date", "@x{%F}"), ("Returns", "@y{0.0000}")])
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

            # Stable filename in the research project root (parent of plots/) for quick open in IDE;
            # versioned file above remains the archival copy.
            main_alias = os.path.join(os.path.dirname(self.plots_dir), "main.html")
            try:
                with open(main_alias, "w", encoding="utf-8") as f:
                    f.write(html_out)
            except OSError:
                pass
            else:
                print(f"Also written to: {main_alias}")
            
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
    
