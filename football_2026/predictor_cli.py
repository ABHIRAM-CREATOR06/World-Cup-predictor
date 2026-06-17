"""Interactive World Cup predictor CLI.

Usage:
    python predictor_cli.py                    # Run predictions from historical data
    python predictor_cli.py --load-results path/to/results.csv  # Load actuals and update predictions
    python predictor_cli.py --add-result DATE HOME TEAM AWAY HSCORE AS  # Add single result

The model automatically incorporates match results and updates predictions.
"""
from datetime import date
from pathlib import Path

import pandas as pd
import argparse

from predictor import WorldCupPredictor, OUT_DIR


def run_predictions(predictor: WorldCupPredictor, ref_date=None, show_backtest=True):
    """Run and display predictions."""
    print("Building predictor...")
    results = predictor.predict_tournament(ref_date)

    if show_backtest and results.backtest_metrics and results.backtest_metrics.get("n_matches", 0) > 0:
        print("\n" + "=" * 70)
        print("BACKTEST METRICS (vs user-provided results)")
        print("=" * 70)
        m = results.backtest_metrics
        print(f"  Matches evaluated:  {m['n_matches']}")
        print(f"  Outcome accuracy:   {m['outcome_accuracy']*100:.1f}%")
        print(f"  Brier score:        {m['brier']:.4f}")
        print(f"  Log loss:           {m['log_loss']:.4f}")

    if results.tournament_probs is not None and len(results.tournament_probs) > 0:
        print("\n" + "=" * 70)
        print("TOP 10 CHAMPIONSHIP PROBABILITIES")
        print("=" * 70)
        for _, row in results.tournament_probs.head(10).iterrows():
            print(f"  {row['P_Win']*100:5.2f}%  {row['team']:25s}  (ELO {row['elo']:.0f})")

    if results.group_match_predictions is not None and len(results.group_match_predictions) > 0:
        print("\n" + "=" * 70)
        print(f"REMAINING GROUP MATCHES ({len(results.group_match_predictions)} unplayed)")
        print("=" * 70)
        for letter in sorted(set(results.group_match_predictions["group"]))[:4]:
            sub = results.group_match_predictions[results.group_match_predictions["group"] == letter]
            for _, r in sub.iterrows():
                print(f"  {letter} {r['home']:20s} vs {r['away']:20s}  "
                      f"P(H/D/A): {r['ph']:.2f}/{r['pd']:.2f}/{r['pa']:.2f}")

    predictor.save_outputs(ref_date)
    print(f"\nOutputs saved to {OUT_DIR}")


def main():
    import sys
    parser = argparse.ArgumentParser(description="World Cup 2026 predictor")
    parser.add_argument("--load-results", type=Path, help="Path to CSV with match results")
    parser.add_argument("--date", type=str, help="Reference date (YYYY-MM-DD)")
    parser.add_argument("--sims", type=int, default=5000, help="Number of Monte Carlo simulations")
    parser.add_argument("--no-backtrack", action="store_true", help="Don't show backtest metrics")
    args = parser.parse_args()

    predictor = WorldCupPredictor(n_simulations=args.sims)

    if args.load_results:
        n = predictor.load_actuals(args.load_results)
        print(f"Loaded {n} results from {args.load_results}")

    ref_date = pd.Timestamp(args.date).date() if args.date else None
    run_predictions(predictor, ref_date, show_backtest=not args.no_backtrack)


if __name__ == "__main__":
    main()