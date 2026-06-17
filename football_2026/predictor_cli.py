"""Interactive World Cup predictor CLI.

Usage:
    python predictor.py                    # Run predictions from historical data
    python predictor.py --load-results     # Load actuals from CSV and update predictions
    python predictor.py --backtest         # Show backtest metrics against loaded results

The model automatically incorporates match results and updates predictions.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from predictor import WorldCupPredictor, OUT_DIR


def run_predictions(predictor: WorldCupPredictor, ref_date=None):
    """Run and display predictions."""
    results = predictor.predict_tournament(ref_date)

    print("\n" + "=" * 70)
    print("TOP 10 CHAMPIONSHIP PROBABILITIES")
    print("=" * 70)
    for _, row in results.tournament_probs.head(10).iterrows():
        print(f"  {row['P_Win']*100:5.2f}%  {row['team']:25s}  (ELO {row['elo']:.0f})")

    if results.backtest_metrics:
        print("\n" + "=" * 70)
        print("BACKTEST METRICS (vs user-provided results)")
        print("=" * 70)
        m = results.backtest_metrics
        print(f"  Matches evaluated:  {m['n_matches']}")
        print(f"  Outcome accuracy:   {m['outcome_accuracy']*100:.1f}%")
        print(f"  Brier score:        {m['brier']:.4f}")
        print(f"  Log loss:           {m['log_loss']:.4f}")

    print("\n" + "=" * 70)
    print(f"REMAINING GROUP MATCHES ({len(results.group_match_predictions)} unplayed)")
    print("=" * 70)
    for letter in sorted(set(results.group_match_predictions["group"])):
        sub = results.group_match_predictions[results.group_match_predictions["group"] == letter]
        for _, r in sub.iterrows():
            print(f"  {letter} {r['home']:20s} vs {r['away']:20s}  "
                  f"P(H/D/A): {r['ph']:.2f}/{r['pd']:.2f}/{r['pa']:.2f}  "
                  f"Likely: {r['most_likely_score']}")

    predictor.save_outputs(ref_date)
    print(f"\nOutputs saved to {OUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="World Cup 2026 predictor")
    parser.add_argument("--load-results", type=Path, help="Path to CSV with match results")
    parser.add_argument("--date", type=str, help="Reference date (YYYY-MM-DD)")
    parser.add_argument("--backtest", action="store_true", help="Show backtest after predictions")
    parser.add_argument("--add-result", nargs=5, metavar=("DATE", "HOME", "AWAY", "HSCORE", "ASCORE"),
                        help="Add a single result: DATE HOME TEAM AWAY HSCORE AS")
    args = parser.parse_args()

    predictor = WorldCupPredictor()

    if args.load_results:
        n = predictor.load_actuals(args.load_results)
        print(f"Loaded {n} results from {args.load_results}")

    if args.add_result:
        d, h, a, hs, as_ = args.add_result
        predictor.add_result(
            match_date=pd.Timestamp(d).date(),
            home_team=h,
            away_team=a,
            home_score=int(hs),
            away_score=int(as_),
        )
        print(f"Added result: {h} {hs}-{as_} {a}")

    ref_date = pd.Timestamp(args.date).date() if args.date else None
    run_predictions(predictor, ref_date)


if __name__ == "__main__":
    main()