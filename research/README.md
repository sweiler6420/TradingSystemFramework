# Research Directory

This directory contains all trading strategy research projects, separated from the core framework to maintain clean architecture.

## Structure

Each research project follows a standardized structure:

```
research/
├── create_project.py          # Entry script to create new projects
├── mach_1/                   # Example project directory
│   ├── README.md            # Project documentation
│   ├── main.py              # Main research script
│   ├── data/                # Raw and processed data
│   ├── results/             # Test results and metrics
│   ├── plots/               # Interactive graphs
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

## Research Test Framework

Each project includes 4 standardized tests:

### 1. In-Sample Excellence Test
- **Purpose:** Proof of concept validation
- **Description:** Test strategy performance on historical data
- **Always implemented first**

### 2. In-Sample Permutation Test  
- **Purpose:** Statistical significance validation
- **Description:** Monte Carlo permutation test to validate results
- **Validates that results aren't due to random chance**

### 3. Walk Forward Test
- **Purpose:** Out-of-sample validation
- **Description:** Rolling window validation
- **Tests robustness across different time periods**

### 4. Walk Forward Permutation Test
- **Purpose:** Out-of-sample statistical validation
- **Description:** Monte Carlo permutation test on walk-forward results
- **Final validation of statistical significance**

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
4. Run the research: `python main.py`
5. Document findings in `notes/` and `README.md`
