# World Cup Predictor

Interactive 2026 FIFA World Cup prediction model with:
- ELO ratings updated as match results are added
- Attack/defense ratings with shrinkage for low-sample teams
- Bivariate Poisson goal model for better scoreline prediction
- Monte Carlo simulation for tournament outcomes

## Usage

### Command Line

```bash
# Run predictions (no actuals loaded)
python predictor_cli.py --sims 5000

# Load actual match results and update predictions
python predictor_cli.py --load-results out/actual_results_2026.csv --sims 5000

# Skip backtest for faster execution
python predictor_cli.py --load-results out/actual_results_2026.csv --skip-backtest
```

### Python API

```python
from predictor import WorldCupPredictor
from datetime import date

# Create predictor
p = WorldCupPredictor(n_simulations=5000)

# Add match results
p.add_result(date(2026, 6, 11), "Mexico", "South Africa", 2, 0)
p.add_result(date(2026, 6, 12), "USA", "Paraguay", 4, 1)

# Get updated predictions
results = p.predict_tournament()
print(results.tournament_probs.head(10))

# Save outputs
p.save_outputs()
```

### GUI

```bash
# Tkinter GUI (no dependencies)
python predictor_gui.py
```

## Accuracy Features

1. **Rating Shrinkage**: Teams with fewer matches get ratings pulled toward average (1500)
2. **Bivariate Poisson**: Accounts for correlation between teams' goal counts
3. **Recency Weighting**: Recent matches weighted more heavily in ratings
4. **Backtesting**: Automatic evaluation against provided results

## Output Files

All saved to `out/`:
- `tournament_probabilities.csv` - Stage progression probabilities
- `group_match_predictions.csv` - Unplayed match predictions
- `group_position_probabilities.csv` - Group position chances
- `elo_snapshot_2026.csv` - Current ELO ratings
- `attack_defense_ratings.csv` - Team attack/defense multipliers
- `summary.json` - Complete summary with metrics