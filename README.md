# ⚽ 2026 FIFA World Cup Predictor

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Advanced predictive model for the 2026 FIFA World Cup with dynamic ELO updates and Monte Carlo simulation.

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd football_2026

# Install dependencies
pip install numpy pandas scipy

# Run predictions
python predictor_cli.py --sims 5000
```

## 📊 Features

| Feature | Description |
|---------|-------------|
| 🔄 **Dynamic Updates** | Add match results and predictions update automatically |
| 📈 **ELO Ratings** | FiveThirtyEight-style ELO with goal-difference adjustments |
| ⚽ **Poisson Model** | Bivariate Poisson for correlated goal scoring |
| 🎯 **Monte Carlo** | 20,000 simulations for tournament probabilities |
| 📊 **Backtesting** | Automatic Brier score and accuracy evaluation |
| 📉 **Shrinkage** | Regularized ratings for teams with limited data |

## 💻 Usage

### Command Line Interface

```bash
# Basic predictions (no actuals)
python predictor_cli.py --sims 5000

# Load actual match results and update predictions
python predictor_cli.py --load-results out/actual_results_2026.csv --sims 5000

# Skip backtest for faster execution
python predictor_cli.py --load-results out/actual_results_2026.csv --skip-backtest

# Use specific reference date
python predictor_cli.py --date 2026-06-20
```

**Options:**
| Flag | Description |
|------|-------------|
| `--load-results PATH` | CSV file with match results to fold in |
| `--date YYYY-MM-DD` | Reference date (defaults to today) |
| `--sims N` | Number of Monte Carlo simulations (default: 3000) |
| `--skip-mc` | Skip tournament simulation |
| `--skip-backtest` | Skip backtest computation |

### Python API

```python
from predictor import WorldCupPredictor
from datetime import date

# Create predictor with 5000 simulations
predictor = WorldCupPredictor(n_simulations=5000)

# Add match results dynamically
predictor.add_result(date(2026, 6, 11), "Mexico", "South Africa", 2, 0)
predictor.add_result(date(2026, 6, 12), "USA", "Paraguay", 4, 1)

# Get predictions
results = predictor.predict_tournament()

# Access top 10 championship probabilities
print(results.tournament_probs.head(10))

# Save all outputs to out/ directory
predictor.save_outputs()
```

### GUI (Tkinter)

```bash
# Run the desktop GUI
python predictor_gui.py
```

Features:
- Load results from CSV
- Add single match results interactively
- View top 10 championship probabilities
- Browse remaining group match predictions
- See backtest metrics

## 📁 Project Structure

```
football_2026/
├── predictor.py          # Main predictor class
├── predictor_cli.py      # Command-line interface
├── predictor_gui.py      # Tkinter desktop GUI
├── main.py               # Original standalone predictor
├── main_with_actuals.py    # Full pipeline with result folding
├── elo.py                # ELO rating system
├── poisson.py            # Attack/defense and goal model
├── simulator.py          # Tournament simulation
├── data.py               # Data loading and normalization
├── backtest.py           # Backtesting utilities
├── fifa_groups.json      # Official group composition
├── out/                  # Output directory
│   ├── tournament_probabilities.csv
│   ├── group_match_predictions.csv
│   ├── elo_snapshot_2026.csv
│   └── summary.json
└── README.md
```

## 📈 Model Details

### ELO System
- Initial rating: 1500
- K-factor: 32 (scaled by tournament importance)
- Home advantage: 95 ELO points
- Goal-difference multiplier for rapid updates

### Goal Scoring Model
- Bivariate Poisson with correlation (ρ = 0.15)
- Recency-weighted (3-year half-life)
- Tournament importance weighting

### Rating Shrinkage
Teams with fewer than 10 matches get 30% shrinkage toward 1500.
Teams with fewer than 25 matches get 15% shrinkage.

This prevents overrating teams from limited competitive match history.

## 📊 Output Files

| File | Description |
|------|-------------|
| `tournament_probabilities.csv` | P(R32/R16/QF/SF/Final/Win) per team |
| `group_match_predictions.csv` | Predictions for unplayed group matches |
| `group_position_probabilities.csv` | P(1st/2nd/3rd/4th/3rd_qual) per team |
| `elo_snapshot_2026.csv` | Current ELO ratings for all WC teams |
| `attack_defense_ratings.csv` | Team attack/defense multipliers |
| `summary.json` | Complete summary with backtest metrics |

## 🎯 Example Output

```
TOP 10 CHAMPIONSHIP PROBABILITIES
  15.7%  Morocco                    (ELO 2020)
  12.4%  Argentina                  (ELO 2200)
   9.1%  England                    (ELO 2088)
   7.8%  Japan                      (ELO 2020)
   6.5%  Spain                      (ELO 2228)
```

## 📊 Backtesting

When actual results are provided via `--load-results`, the model evaluates:
- Outcome accuracy (correct H/D/A prediction)
- Brier score (lower is better)
- Log loss (lower is better)
- Goal prediction MAE

## 🔧 Configuration

Edit `predictor.py` to adjust:
- `RHO` - Goal correlation parameter (default: 0.15)
- `min_matches` in `_estimate_ratings_improved()` - Shrinkage threshold
- `n_simulations` - Monte Carlo iterations

## 📦 Requirements

```
numpy
pandas
scipy
```

For GUI:
```
tkinter (included with Python)
```

## 🏆 Bracket

The model uses the official FIFA 2026 bracket with 48 teams in 12 groups.
R32 features 8 best third-place teams assigned via bipartite matching.

## 📜 License

MIT License