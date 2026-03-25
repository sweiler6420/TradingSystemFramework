"""
Cross-symbol summary for a single versioned run folder (e.g. ``results/V0003/``).

Each row is one **independent** in-sample backtest. Aggregate rows are **unweighted**
cross-sectional summaries (mean / median / dispersion), not a combined portfolio
or a single merged equity curve.
"""

from __future__ import annotations

import html as html_module
import json
import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl


@dataclass(frozen=True)
class MetricSpec:
    key: str
    header: str
    is_pct: bool
    decimals: int


PERFORMANCE_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("profit_factor", "Profit factor", False, 4),
    MetricSpec("sharpe_ratio", "Sharpe", False, 4),
    MetricSpec("sortino_ratio", "Sortino", False, 4),
    MetricSpec("max_drawdown", "Max DD", True, 2),
    MetricSpec("total_return", "Tot. return", True, 2),
    MetricSpec("win_rate", "Win rate", True, 2),
)

BATCH_MD_NAME = "insample_excellence_batch_summary.md"
BATCH_JSON_NAME = "insample_excellence_batch_summary.json"
BATCH_HTML_NAME = "insample_excellence_batch_summary.html"
METADATA_GLOB = "insample_excellence_*_metadata.json"

# Dark report chrome — align with ``bokeh_interactive_plot_creator.py`` (static HTML, no Bokeh import).
_HTML_PAGE_BG = "#0a0f18"
_HTML_CARD_BG = "#0e1522"
_HTML_GRID = "#1c2838"
_HTML_AXIS_TEXT = "#7a8494"
_HTML_TITLE_TEXT = "#8b95a6"
_HTML_METRIC_VALUE = "#7d8796"
_HTML_CORNER_RADIUS = 10


def _finite(x: float) -> bool:
    return math.isfinite(x)


def parse_data_period_ends(data_period: str) -> tuple[str, str]:
    """Split ``data_period`` like ``\"2024-03-24 22:00:00 to 2026-02-27 21:00:00\"``."""
    s = (data_period or "").strip()
    if " to " in s:
        a, b = s.split(" to ", 1)
        return a.strip(), b.strip()
    return s, ""


def display_ticker(meta: dict[str, Any]) -> str:
    sym = meta.get("symbol")
    if isinstance(sym, str) and sym.strip():
        return sym.strip()
    tn = meta.get("test_name") or ""
    slug = ""
    if isinstance(tn, str) and tn.startswith("insample_excellence_"):
        slug = tn[len("insample_excellence_") :]
    else:
        slug = str(tn or "unknown")
    # Legacy runs without ``symbol``: slug like ``C_EURUSD`` → ``C:EURUSD`` (Massive-style FX prefix).
    m = re.match(r"^([A-Za-z0-9])_(.+)$", slug)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    return slug


def _fmt_metric(spec: MetricSpec, value: float) -> str:
    if not _finite(value):
        return "—"
    if spec.is_pct:
        return f"{100.0 * value:.{spec.decimals}f}%"
    return f"{value:.{spec.decimals}f}"


def _row_dict(meta: dict[str, Any]) -> dict[str, Any]:
    perf = meta.get("performance_results") or {}
    if not isinstance(perf, dict):
        perf = {}
    first_bar, last_bar = parse_data_period_ends(str(meta.get("data_period") or ""))

    out: dict[str, Any] = {
        "ticker": display_ticker(meta),
        "test_name": meta.get("test_name"),
        "bars": meta.get("data_points"),
        "first_bar": first_bar,
        "last_bar": last_bar,
    }
    for spec in PERFORMANCE_METRICS:
        v = perf.get(spec.key)
        try:
            fv = float(v) if v is not None else float("nan")
        except (TypeError, ValueError):
            fv = float("nan")
        out[spec.key] = fv
    return out


def _aggregate(values: list[float]) -> dict[str, float | None]:
    xs = [x for x in values if _finite(x)]
    n = len(xs)
    if n == 0:
        return {"n": 0.0, "mean": float("nan"), "median": float("nan"), "stdev": None}
    mean_v = statistics.fmean(xs)
    med_v = float(statistics.median(xs))
    st_v: float | None
    if n >= 2:
        st_v = statistics.stdev(xs)
    else:
        st_v = None
    return {"n": float(n), "mean": mean_v, "median": med_v, "stdev": st_v}


def compute_equal_weight_portfolio_metrics(
    return_frames: list[pl.DataFrame],
) -> dict[str, Any] | None:
    """
    Align per-asset strategy return series on shared ``timestamp`` values; each bar uses the
    arithmetic mean of those returns (equal risk budget per asset, full binary exposure per name).

    Returns ``None`` if fewer than two frames or no shared timestamps remain.
    """
    if len(return_frames) < 2:
        return None

    from framework.performance import (
        MaxDrawdownMeasure,
        ProfitFactorMeasure,
        SharpeRatioMeasure,
        SortinoRatioMeasure,
        TotalReturnMeasure,
        WinRateMeasure,
    )

    acc = return_frames[0]
    for f in return_frames[1:]:
        acc = acc.join(f, on="timestamp", how="inner")

    if acc.height < 2:
        return None

    ret_cols = [c for c in acc.columns if c.startswith("ret_")]
    if len(ret_cols) < 2:
        return None

    portfolio_r = acc.select(
        (pl.sum_horizontal([pl.col(c) for c in ret_cols]) / float(len(ret_cols))).alias(
            "_port_r"
        )
    )["_port_r"]

    measures = {
        "profit_factor": ProfitFactorMeasure(),
        "sharpe_ratio": SharpeRatioMeasure(),
        "sortino_ratio": SortinoRatioMeasure(),
        "max_drawdown": MaxDrawdownMeasure(),
        "total_return": TotalReturnMeasure(),
        "win_rate": WinRateMeasure(),
    }
    perf = {name: float(m.calculate(portfolio_r)) for name, m in measures.items()}

    ts0 = acc["timestamp"][0]
    ts1 = acc["timestamp"][-1]

    return {
        "n_assets": len(ret_cols),
        "n_bars": int(acc.height),
        "first_bar": str(ts0),
        "last_bar": str(ts1),
        "performance_results": perf,
        "method_note": (
            "Equal notional per asset; each bar is the mean of aligned strategy returns."
        ),
        "method": (
            "At each timestamp we take the arithmetic mean of strategy returns across assets "
            "that share that bar. Assumes equal notional and full binary exposure "
            "per asset each bar — a simple diversified path for comparison to single-name rows, "
            "not an optimized portfolio or volatility target."
        ),
    }


def build_batch_payload(
    run_dir: str | Path,
    records: list[dict[str, Any]],
    *,
    strategy_name: str | None = None,
    equal_weight_portfolio: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structure used for JSON export and markdown rendering."""
    rd = Path(run_dir).resolve()
    rows = [_row_dict(m) for m in records]
    sn = strategy_name
    if not sn and records:
        s0 = records[0].get("strategy_name")
        if isinstance(s0, str):
            sn = s0

    agg: dict[str, Any] = {"n_underlyings": len(rows)}
    for spec in PERFORMANCE_METRICS:
        vals: list[float] = []
        for r in rows:
            if spec.key not in r:
                continue
            try:
                fv = float(r[spec.key])
            except (TypeError, ValueError):
                continue
            if _finite(fv):
                vals.append(fv)
        agg[spec.key] = _aggregate(vals)

    dds = [
        float(r["max_drawdown"])
        for r in rows
        if _finite(float(r.get("max_drawdown", float("nan"))))
    ]
    if dds:
        agg["max_drawdown_worst"] = min(dds)
        agg["max_drawdown_best"] = max(dds)
    else:
        agg["max_drawdown_worst"] = None
        agg["max_drawdown_best"] = None

    trs = [
        float(r["total_return"])
        for r in rows
        if _finite(float(r.get("total_return", float("nan"))))
    ]
    if trs:
        agg["total_return_best"] = max(trs)
        agg["total_return_worst"] = min(trs)
    else:
        agg["total_return_best"] = None
        agg["total_return_worst"] = None

    interp_parts = [
        "Each row is **one market at a time** with **full binary exposure** in that market "
        "(long / short / flat). The first summary block averages **end-of-run metrics** across "
        "those independent paths — it is **not** one merged account or optimized allocation."
    ]
    if equal_weight_portfolio:
        interp_parts.append(
            "The **equal-weight blended** block uses the bar-by-bar **mean** of strategy returns on "
            "timestamps where **all** underlyings align, with equal notional per asset. "
            "Use it to see whether naive diversification would have mattered much versus scanning "
            "single-name results alone."
        )

    return {
        "run_dir": str(rd),
        "run_version": str(records[0].get("run_version", "")) if records else "",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy_name": sn,
        "interpretation": " ".join(interp_parts),
        "rows": rows,
        "aggregate": agg,
        "equal_weight_portfolio": equal_weight_portfolio,
    }


def _html_metric_blocks_from_perf(perf: dict[str, Any]) -> str:
    """Same tile layout as the top summary card; values from a ``performance_results`` dict."""
    blocks: list[str] = []
    for spec in PERFORMANCE_METRICS:
        v = perf.get(spec.key)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = float("nan")
        display = _fmt_metric(spec, fv) if _finite(fv) else "—"
        label = html_module.escape(spec.header.upper())
        blocks.append(
            f'<div class="metric-item">'
            f'<span class="metric-label">{label}</span>'
            f'<span class="metric-value">{html_module.escape(display)}</span>'
            f"</div>"
        )
    return "".join(blocks)


def _summary_metric_cells(agg: dict[str, Any], field: str) -> list[str]:
    cells: list[str] = []
    for spec in PERFORMANCE_METRICS:
        block = agg.get(spec.key)
        if isinstance(block, dict):
            v = block.get(field)
            if isinstance(v, (int, float)) and _finite(float(v)):
                cells.append(_fmt_metric(spec, float(v)))
                continue
        cells.append("—")
    return cells


def render_batch_markdown(payload: dict[str, Any]) -> str:
    rows = payload["rows"]
    specs = PERFORMANCE_METRICS
    headers = ["Ticker", "Bars", "First bar", "Last bar"] + [s.header for s in specs]
    sep = ["---"] * len(headers)

    lines = [
        "# In-sample excellence — batch summary",
        "",
        f"**Run:** `{payload.get('run_version', '')}`  ",
        f"**Strategy:** {payload.get('strategy_name') or '—'}  ",
        f"**Generated (UTC):** {payload.get('generated_at', '')}  ",
        "",
        "## How to read this",
        "",
        payload.get("interpretation", ""),
        "",
        "## Per-underlying results",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]

    for r in rows:
        cells = [
            str(r["ticker"]),
            str(r.get("bars", "")),
            str(r.get("first_bar", "")),
            str(r.get("last_bar", "")),
        ]
        for spec in specs:
            cells.append(_fmt_metric(spec, float(r[spec.key])))
        lines.append("| " + " | ".join(cells) + " |")

    agg = payload["aggregate"]
    n = int(agg.get("n_underlyings", 0))

    mean_cells = _summary_metric_cells(agg, "mean")
    med_cells = _summary_metric_cells(agg, "median")
    lines.append("")
    lines.append("## Cross-sectional aggregates (unweighted across underlyings)")
    lines.append("")
    lines.append("| **Mean** | — | — | — | — | " + " | ".join(mean_cells) + " |")
    lines.append("| **Median** | — | — | — | — | " + " | ".join(med_cells) + " |")

    if n >= 2:
        sd_cells = _summary_metric_cells(agg, "stdev")
        lines.append(
            "| **Stdev (across names)** | — | — | — | — | " + " | ".join(sd_cells) + " |"
        )

    ewp = payload.get("equal_weight_portfolio")
    if isinstance(ewp, dict) and ewp.get("performance_results"):
        pr = ewp["performance_results"]
        lines.extend(
            [
                "",
                "## Equal-weight blended path",
                "",
                f"Bars: **{ewp.get('n_bars')}** · Assets: **{ewp.get('n_assets')}** · "
                f"First bar → last: `{ewp.get('first_bar', '')}` → `{ewp.get('last_bar', '')}`",
                "",
            ]
        )
        for spec in specs:
            v = pr.get(spec.key)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float("nan")
            lines.append(f"- **{spec.header}:** {_fmt_metric(spec, fv)}")
        lines.extend(["", str(ewp.get("method", "")), ""])

    dd_w = agg.get("max_drawdown_worst")
    dd_b = agg.get("max_drawdown_best")
    tr_b = agg.get("total_return_best")
    tr_w = agg.get("total_return_worst")
    lines.extend(
        [
            "",
            "## Cohort extremes (best / worst single-underlying outcomes)",
            "",
        ]
    )
    _dd_spec = MetricSpec("max_drawdown", "Max DD", True, 2)
    _tr_spec = MetricSpec("total_return", "Tot. return", True, 2)
    if isinstance(dd_w, (int, float)) and _finite(float(dd_w)):
        lines.append(
            f"- **Worst max drawdown (among underlyings):** {_fmt_metric(_dd_spec, float(dd_w))}"
        )
    if isinstance(dd_b, (int, float)) and _finite(float(dd_b)):
        lines.append(
            f"- **Least severe max drawdown (among underlyings):** {_fmt_metric(_dd_spec, float(dd_b))}"
        )
    if isinstance(tr_b, (int, float)) and _finite(float(tr_b)):
        lines.append(
            f"- **Best total return (among underlyings):** {_fmt_metric(_tr_spec, float(tr_b))}"
        )
    if isinstance(tr_w, (int, float)) and _finite(float(tr_w)):
        lines.append(
            f"- **Worst total return (among underlyings):** {_fmt_metric(_tr_spec, float(tr_w))}"
        )

    return "\n".join(lines)


def render_batch_html(
    payload: dict[str, Any],
    *,
    bokeh_resources_html: str = "",
    bokeh_chart_html: str = "",
) -> str:
    """Standalone dark HTML: top summary bar (cross-sectional mean), optional Bokeh equity block, table."""
    bg = _HTML_PAGE_BG
    card = _HTML_CARD_BG
    grid = _HTML_GRID
    axis = _HTML_AXIS_TEXT
    title_c = _HTML_TITLE_TEXT
    val_c = _HTML_METRIC_VALUE
    r = _HTML_CORNER_RADIUS

    run_v = html_module.escape(str(payload.get("run_version") or ""))
    strat = html_module.escape(str(payload.get("strategy_name") or "—"))
    gen = html_module.escape(str(payload.get("generated_at") or ""))
    n_u = int((payload.get("aggregate") or {}).get("n_underlyings") or 0)
    agg = payload.get("aggregate") or {}

    # Performance summary bar — same metric order as interactive reports; values = cohort mean.
    metric_blocks: list[str] = []
    for spec in PERFORMANCE_METRICS:
        block = agg.get(spec.key)
        mean_v: float | None = None
        if isinstance(block, dict) and block.get("mean") is not None:
            try:
                mean_v = float(block["mean"])
            except (TypeError, ValueError):
                mean_v = None
        display = _fmt_metric(spec, mean_v) if mean_v is not None and _finite(mean_v) else "—"
        label = html_module.escape(spec.header.upper())
        metric_blocks.append(
            f'<div class="metric-item">'
            f'<span class="metric-label">{label}</span>'
            f'<span class="metric-value">{html_module.escape(display)}</span>'
            f"</div>"
        )

    rows = payload.get("rows") or []
    header_cells = (
        ["Ticker", "Bars", "First bar", "Last bar"]
        + [spec.header for spec in PERFORMANCE_METRICS]
    )
    ths = "".join(
        f"<th>{html_module.escape(h)}</th>" for h in header_cells
    )

    trs: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cells = [
            html_module.escape(str(row.get("ticker", ""))),
            html_module.escape(str(row.get("bars", ""))),
            html_module.escape(str(row.get("first_bar", ""))),
            html_module.escape(str(row.get("last_bar", ""))),
        ]
        for spec in PERFORMANCE_METRICS:
            v = row.get(spec.key)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float("nan")
            cells.append(html_module.escape(_fmt_metric(spec, fv)))
        trs.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    tbody = "\n".join(trs)

    ext_lines: list[str] = []
    ewp = payload.get("equal_weight_portfolio")
    blend_section = ""
    if isinstance(ewp, dict) and ewp.get("performance_results"):
        perf = ewp["performance_results"]
        if isinstance(perf, dict):
            nb = html_module.escape(str(ewp.get("n_bars", "")))
            na = html_module.escape(str(ewp.get("n_assets", "")))
            fb = html_module.escape(str(ewp.get("first_bar", "")))
            lb = html_module.escape(str(ewp.get("last_bar", "")))
            meth_short = html_module.escape(
                str(ewp.get("method_note") or ewp.get("method") or "")
            )
            blend_blocks = _html_metric_blocks_from_perf(perf)
            blend_section = f"""
    <section class="summary-card" aria-label="Equal-weight blended metrics">
      <div class="card-kicker">Equal-weight blended timeline</div>
      <div class="card-sub">
        {nb} bars · {na} underlyings · {fb} → {lb}. {meth_short}
      </div>
      <div class="metric-grid">
        {blend_blocks}
      </div>
    </section>
"""

    for label, key in (
        ("Worst max drawdown (among underlyings)", "max_drawdown_worst"),
        ("Least severe max drawdown (among underlyings)", "max_drawdown_best"),
        ("Best total return (among underlyings)", "total_return_best"),
        ("Worst total return (among underlyings)", "total_return_worst"),
    ):
        raw = agg.get(key)
        if isinstance(raw, (int, float)) and _finite(float(raw)):
            if "drawdown" in key:
                spec = MetricSpec("max_drawdown", "", True, 2)
            else:
                spec = MetricSpec("total_return", "", True, 2)
            ext_lines.append(
                f"<li><strong>{html_module.escape(label)}:</strong> "
                f"{html_module.escape(_fmt_metric(spec, float(raw)))}</li>"
            )
    cohort_html = ""
    if ext_lines:
        cohort_html = (
            f'<section class="cohort-section">'
            f'<h2>Cohort extremes <span style="font-weight:400;text-transform:none;letter-spacing:0;color:{_HTML_AXIS_TEXT};font-size:0.92em;">(best / worst single-underlying outcomes)</span></h2><ul>{"".join(ext_lines)}</ul></section>'
        )

    interp = html_module.escape(str(payload.get("interpretation") or ""))

    has_bokeh = bool(bokeh_chart_html.strip())
    body_cls = "has-bokeh-embed" if has_bokeh else ""
    bokeh_embed_css = ""
    if has_bokeh:
        bokeh_embed_css = f"""
    body.has-bokeh-embed {{
      --background-color: {card} !important;
      --tooltip-text: {axis};
      --tooltip-color: #121b2c;
      --tooltip-border: {grid};
    }}
    :root {{
      --background-color: {card} !important;
    }}
"""
    equity_section = ""
    if has_bokeh:
        equity_section = f"""
    <section class="summary-card equity-chart" aria-label="Batch equity comparison">
      <div class="card-kicker">Equity comparison</div>
      <div class="card-sub">
        Interactive chart — use the legend to hide series. Buy &amp; hold curves use muted tints of the same hues as strategy.
        With several underlyings, thick lines show equal-weight blends of strategy vs buy &amp; hold (combined vs combined).
      </div>
      {bokeh_chart_html}
    </section>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>In-sample batch summary — {run_v}</title>
  {bokeh_resources_html}
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 28px 20px 40px;
      min-height: 100%;
      background: {bg};
      color: {axis};
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 15px;
      line-height: 1.45;
    }}
    .page {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 1.35rem;
      font-weight: 600;
      color: {title_c};
      letter-spacing: 0.02em;
    }}
    .meta-line {{
      margin: 0 0 22px;
      font-size: 0.88rem;
      color: {axis};
    }}
    .meta-line strong {{ color: {title_c}; font-weight: 600; }}
    .summary-card {{
      background: {card};
      border-radius: {r}px;
      padding: clamp(20px, 3vh, 36px) clamp(18px, 4vw, 40px);
      margin-bottom: 26px;
      box-shadow: 0 1px 0 {grid};
    }}
    .summary-card .card-kicker {{
      color: {title_c};
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      margin-bottom: 14px;
    }}
    .summary-card .card-sub {{
      color: {axis};
      font-size: 12px;
      margin: -8px 0 18px;
      letter-spacing: 0.02em;
    }}
    .metric-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 18px 32px;
      align-items: flex-start;
    }}
    .metric-item {{
      min-width: 140px;
    }}
    .metric-label {{
      display: block;
      color: {title_c};
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
    }}
    .metric-value {{
      display: block;
      color: {val_c};
      font-size: 15px;
      font-variant-numeric: tabular-nums;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 0.95rem;
      font-weight: 600;
      color: {title_c};
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .table-wrap {{
      background: {card};
      border-radius: {r}px;
      padding: 20px 20px 12px;
      overflow-x: auto;
      margin-bottom: 24px;
      border: 1px solid {grid};
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }}
    th {{
      text-align: left;
      padding: 12px 14px 12px 0;
      border-bottom: 1px solid {grid};
      color: {title_c};
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      white-space: nowrap;
    }}
    td {{
      padding: 12px 14px 12px 0;
      border-bottom: 1px solid {grid};
      color: {axis};
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: rgba(28, 40, 56, 0.35); }}
    .cohort-section ul {{
      margin: 0;
      padding-left: 1.2rem;
      color: {axis};
      font-size: 14px;
    }}
    .cohort-section li {{ margin-bottom: 8px; }}
    .footnote {{
      margin-top: 18px;
      padding-top: 16px;
      border-top: 1px solid {grid};
      font-size: 12px;
      color: {axis};
      max-width: 72ch;
    }}
    .equity-chart .bk-root {{
      margin-top: 8px;
    }}
{bokeh_embed_css}
  </style>
</head>
<body class="{body_cls}">
  <div class="page">
    <h1>In-sample excellence — batch summary</h1>
    <p class="meta-line">
      <strong>Run</strong> {run_v} &nbsp;·&nbsp;
      <strong>Strategy</strong> {strat} &nbsp;·&nbsp;
      <strong>Underlyings</strong> {n_u} &nbsp;·&nbsp;
      <strong>Generated</strong> {gen}
    </p>

    <section class="summary-card" aria-label="Cross-sectional mean metrics">
      <div class="card-kicker">Performance summary</div>
      <div class="card-sub">
        Cross-sectional <strong>mean</strong> of each metric across underlyings
        (each run = full binary exposure in one market). This is <strong>not</strong> one combined account.
      </div>
      <div class="metric-grid">
        {"".join(metric_blocks)}
      </div>
    </section>

    {blend_section}

    {equity_section}

    <section class="table-section">
      <h2>Per-underlying results</h2>
      <div class="table-wrap">
        <table>
          <thead><tr>{ths}</tr></thead>
          <tbody>
            {tbody}
          </tbody>
        </table>
      </div>
    </section>

    {cohort_html}

    <p class="footnote">{interp}</p>
  </div>
</body>
</html>
"""


def write_insample_batch_summary(
    run_dir: str | Path,
    records: list[dict[str, Any]],
    *,
    strategy_name: str | None = None,
    equal_weight_portfolio: dict[str, Any] | None = None,
    equity_figure: Any | None = None,
) -> tuple[Path, Path, Path]:
    """
    Write ``insample_excellence_batch_summary.md``, ``.json``, and ``.html`` under ``run_dir``.

    ``records`` should be the in-memory metadata dicts from each per-symbol run
    (must include ``performance_results``).

    When ``equity_figure`` is a Bokeh ``figure``, it is embedded in the HTML via ``components()``;
    pass ``None`` when regenerating summaries from disk (no aligned return frames available).
    """
    rd = Path(run_dir).resolve()
    rd.mkdir(parents=True, exist_ok=True)
    payload = build_batch_payload(
        rd,
        records,
        strategy_name=strategy_name,
        equal_weight_portfolio=equal_weight_portfolio,
    )
    md_path = rd / BATCH_MD_NAME
    json_path = rd / BATCH_JSON_NAME
    html_path = rd / BATCH_HTML_NAME

    bokeh_resources_html = ""
    bokeh_chart_html = ""
    if equity_figure is not None:
        from bokeh.embed import components
        from bokeh.resources import CDN

        script, div = components(equity_figure)
        bokeh_resources_html = CDN.render()
        bokeh_chart_html = f"{div}\n{script}"

    md_path.write_text(render_batch_markdown(payload), encoding="utf-8")
    html_path.write_text(
        render_batch_html(
            payload,
            bokeh_resources_html=bokeh_resources_html,
            bokeh_chart_html=bokeh_chart_html,
        ),
        encoding="utf-8",
    )

    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, float) and not _finite(obj):
            return None
        return obj

    json_path.write_text(
        json.dumps(_json_safe(payload), indent=2, default=str),
        encoding="utf-8",
    )
    return md_path, json_path, html_path


def load_insample_metadata_from_run_dir(run_dir: str | Path) -> list[dict[str, Any]]:
    """
    Load per-underlying metadata JSON files from a version folder (for regeneration).

    Skips batch summary files and any non-matching names.
    """
    rd = Path(run_dir).resolve()
    if not rd.is_dir():
        raise FileNotFoundError(f"Not a directory: {rd}")
    out: list[dict[str, Any]] = []
    for p in sorted(rd.glob(METADATA_GLOB)):
        if p.name == BATCH_JSON_NAME:
            continue
        if "batch_summary" in p.name:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("performance_results"):
            out.append(data)
    return out


def regenerate_batch_summary(run_dir: str | Path) -> tuple[Path, Path, Path] | None:
    """Rebuild batch summary from on-disk metadata files. Returns paths if any rows."""
    records = load_insample_metadata_from_run_dir(run_dir)
    if not records:
        return None
    return write_insample_batch_summary(run_dir, records)
