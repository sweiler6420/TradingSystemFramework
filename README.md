# Trading System Framework

A Python framework for **feature-based strategies**, **Polars** OHLCV pipelines, **cached market data** (yfinance, Massive/Polygon, etc.), and **layered research validation** (in-sample runs first; out-of-sample and statistical tests as you grow the suite).

For class-level architecture (features → strategies → backtest), see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## Quick start

```bash
git clone https://github.com/sweiler6420/TradingSystemFramework.git
cd TradingSystemFramework
```

Install with **[uv](https://docs.astral.sh/uv/)** (uses **pyproject.toml** and **uv.lock** for reproducible deps):

```bash
uv sync
```

That creates or updates `.venv` and installs the project in editable mode. Run tools without activating the venv:

```bash
uv run python research/mach4_ema_band_ep1/main.py
# or: uv run python research/run_project.py 4
```

Or activate the venv and use `python` as usual:

```bash
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

**Without uv:** `python -m venv .venv`, activate it, then `pip install -e .` (same dependencies, no lockfile guarantee).

Dependencies include **Polars**, **Bokeh**, **yfinance**, **massive** (Polygon/Massive REST client), etc.

---

## Repository layout

| Area | Role |
|------|------|
| `framework/` | Core library: `DataHandler`, features (`EmaFeature`, …), `SignalBasedStrategy`, `risk_reward`, performance **measures**, significance-testing primitives, `StrategyBacktest`, pluggable `data_sources` + Parquet cache |
| `research/` | One folder per idea (e.g. `mach4_ema_band_ep1/`): `main.py`, `strategies/`, `data/`, `results/`, `plots/`, `tests/config.py` |
| `research/suites/` | Shared **suites** (e.g. **`insample_excellence`**: metrics + Bokeh reports + versioned metadata) |

See **[research/README.md](research/README.md)** for how research projects are structured and how to add one.

---

## Running a research example

Example: **mach4** EMA band project loads EUR/USD 1h data via Massive, runs the in-sample excellence path, and writes plots/results under the project.

```bash
set MASSIVE_API_KEY=your_key   # Windows; use export on Unix
python research/mach4_ema_band_ep1/main.py
```

Adjust symbols, dates, and providers inside that project’s `main.py` and `tests/config.py`.

---

## Testing & validation (how the pieces fit)

### Why “in-sample excellence” is the default right now

- **End-to-end plumbing first**: You need a reliable path—data → signals → returns → **descriptive metrics** (Sharpe, drawdown, …) → plots and saved artifacts—before you trust more advanced steps.
- **Avoid premature optimization**: Grid search and permutation tests on the same slice you tuned **inflates false discovery**. The codebase is staged so you can **prove the strategy runs and reports metrics** on a clean window, then add **OOS** and **significance** in a controlled order.
- **Config placeholders**: Each research package’s `tests/config.py` already sketches **insample / permutation / walk-forward** toggles; **`InSampleExcellenceSuite`** is the one path fully wired today.

---

### A well-rounded research pipeline (what you *could* or *should* run)

Think in **layers** instead of one giant “test”:

1. **Descriptive backtest (in-sample)**  
   One or more **fixed** windows on **historical** data. Report **performance measures** (Sharpe, Sortino, profit factor, max drawdown, …). Goal: **Does the implementation behave?** Any “excellence” threshold here is **engineering + sanity**, not proof of edge.

2. **Parameter exploration (optional, still in-sample)**  
   Many runs over **parameter grids** on the **same** in-sample period. Goal: **Sensitivity**. Treat this as **hypothesis generation**; it is **not** validation unless you hold out data and correct for multiple testing.

3. **Inferential tests on in-sample** (e.g. **permutation / Monte Carlo** on **returns or trades**)  
   Goal: **Is the observed metric plausibly better than random reorderings of the same data?** Answers **“is this luck on this slice?”** — not **“will it work next year?”**

4. **Out-of-sample (holdout or walk-forward)**  
   Evaluate **frozen** rules on **dates the strategy never saw during tuning** (or walk train/test windows). Goal: **Generalization**. Report the **same measures**; interpretation is **stability**, not p-values.

5. **Inferential tests on out-of-sample** (optional)  
   Permutation on **OOS returns** can still be done; interpretation is **“is OOS performance distinguishable from noise on that short window?”** — often **noisy** if OOS is short. **Walk-forward** with many windows is usually more informative than one OOS permutation.

6. **Monte Carlo as *simulation*** (not the same as permutation)  
   Stress paths, position sizing, or model risk. Often a **separate** module from “one number Sharpe ratio.”

---

### Monte Carlo / permutation: in-sample, out-of-sample, or both?

| Question | Where it usually belongs |
|----------|---------------------------|
| “Is this metric **surprising** vs random **labels** on **this** history?” | **In-sample** (or each WF window’s test segment) **permutation** — **significance test**, not a “performance measure.” |
| “Does the **rule** work **on new dates**?” | **Out-of-sample** or **walk-forward** — compare **measures** (Sharpe, DD, etc.) to a **benchmark** or **minimum** you care about. |
| “Is **OOS** performance **lucky**?” | Permutation **on OOS returns** is possible but often **weak** if OOS is short; **many** OOS windows (WF) or **pre-registration** of rules matters more. |

**Do not** mix up: **Sharpe ratio** = descriptive statistic on **one** realized path. **Permutation p-value** = **inferential** statement about **that** path under a **null** (e.g. shuffle returns). They answer different questions.

---

### Performance measures vs significance tests vs “suites”

To keep the system **open-ended**:

| Layer | Examples | Role |
|-------|-----------|------|
| **Performance measures** | Sharpe, Sortino, max drawdown, profit factor | Map **one** backtest path → **numbers** (implement `BaseMeasure`). |
| **Significance / robustness tests** | Monte Carlo permutation, bootstrap, deflated Sharpe, White’s reality check | Map **returns + optional labels** → **p-values, intervals, flags** (see `framework/significance_testing/`). |
| **Evaluation suites** | “In-sample excellence”, “walk-forward report”, “parameter grid” | **Orchestration**: which dates, which measures, which tests, **output** paths — often driven by **research `tests/config.py`**. |

Monte Carlo **permutation** belongs in the **second** row (or a dedicated `robustness/` package), **not** next to Sortino as if it were a third “ratio.” The **suite** can **list** both measures and tests in one JSON config, but the **code** should keep **measure** vs **test** types separate.

---

### What is missing / next steps to wire this up

- **`tests/config.py` is not fully authoritative yet**: `PERFORMANCE_MEASURES` and `TEST_CONFIG` flags are **not** all consumed by `InSampleExcellenceSuite` (measures are still **hardcoded** in the suite class). **Missing**: load measures from config, branch on `insample_permutation` / `walk_forward` / etc.
- **Parameter grid “optimization”**: Not a first-class pipeline yet; you’d add a runner that loops strategies, tags runs, and **writes** separate result folders (and optionally **nested** CV / OOS).
- **Walk-forward**: `TEST_CONFIG["walk_forward"]` is a placeholder until a runner exists that **splits** `DataHandler` by date and aggregates metrics.
- **Naming**: Consider renaming or splitting `framework/performance/monte_carlo_measures.py` vs `significance_testing/` so “Monte Carlo” always reads as **statistical test** or **simulation**, not “another Sharpe.”

---

## License

MIT — see [LICENSE](LICENSE) if present.
