# Research Directory

This directory contains all trading strategy research projects, separated from the core framework to maintain clean architecture.

## Structure

Each research project follows a standardized structure:

```
research/
├── create_project.py          # Entry script to create new projects
├── suites/                    # Shared evaluation suites (insample_excellence, …)
├── mach_1/                   # Example project directory
│   ├── README.md            # Project documentation
│   ├── main.py              # Main research script
│   ├── data/                # Raw and processed data
│   ├── results/             # Versioned runs (V0001/, …): metrics, reports, HTML plots
│   ├── notes/               # Research notes
│   ├── strategies/          # Project-specific strategies
│   ├── tests/               # Test scripts and configs
│   └── archive/             # Archived results
└── mach_2/                   # Another example project
    └── ...
```

## Creating a New Research Project

Use the entry script to create a new project:

```bash
# Create a new project
python research/create_project.py "mach_1" -d "RSI Mean Reversion Strategy"

# Or with just a name
python research/create_project.py "rsi_breakout"
```

This will create:
- ✅ Project directory with standardized structure
- ✅ README.md with project documentation
- ✅ main.py with research script template
- ✅ tests/config.py with test configuration
- ✅ All necessary subdirectories

## Research test framework

Each project’s `tests/config.py` describes **which** validation stages to run (in-sample excellence, permutation, walk-forward, etc.) and **which performance measures** to report.

**Currently:** the shared **`InSampleExcellenceSuite`** (`research/suites/insample_excellence/`) is the main wired path—descriptive metrics + Bokeh plots + versioned metadata. Other toggles in `TEST_CONFIG` are **staged for future runners**; see the root **[README.md](../README.md)** section *Testing & validation* for how measures, significance tests, and suites should stay separate.

### Intended stages (roadmap)

1. **In-sample excellence** — Proof-of-concept: strategy runs, measures computed, plots saved (*implemented*).
2. **In-sample permutation** — Inferential check on the same window (Monte Carlo / shuffle null); **not** a replacement for OOS.
3. **Walk-forward** — Rolling train/test or fixed holdout for **generalization**.
4. **Walk-forward + permutation** — Optional significance on **test** segments per window (interpret carefully).

## Benefits

- 🔄 **Reusable Framework:** Core framework stays clean
- 📊 **Organized Research:** Each idea has its own space
- 📈 **Trackable Progress:** Clear test progression
- 📝 **Documentation:** Standardized notes and findings
- 🔍 **Reproducible:** Complete research record
- 📁 **Archivable:** Easy to archive completed research

## Getting Started

1. Create a new project: `python research/create_project.py "your_idea"`
2. Navigate to the project: `cd research/your_idea`
3. Edit `main.py` to implement your strategy
4. Run the research: `python main.py` **or** from the repo root use the launcher (by mach number or full folder name):

```bash
uv run python research/run.py 4
uv run python research/run.py mach4_ema_band_ep1
```

If two folders exist (`mach4_foo` and `mach4_bar`), the numeric form is ambiguous — pass the full directory name.

5. Document findings in `notes/` and `README.md`
