"""Run backtest on past tournaments and save a summary table.

We use data strictly before each tournament's start to predict all its matches.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\abhir\.mavis\sessions\mvs_fbdb1a4d0ee34fc9b22f88407249a52a\workspace\football_2026")

from data import load_results, load_former_names, build_name_map, normalize_teams
from elo import compute_elo
from poisson import estimate_ratings, match_outcome_probs
from backtest import brier, log_loss, rps

OUT = Path(r"C:\Users\abhir\.mavis\sessions\mvs_fbdb1a4d0ee34fc9b22f88407249a52a\workspace\football_2026\out")

r = load_results()
fn = load_former_names()
nm = build_name_map(fn)
r = normalize_teams(r, nm, fn)

TOURNAMENTS = [
    ("FIFA World Cup", pd.Timestamp("2022-11-20"), "2022 World Cup"),
    ("UEFA Euro", pd.Timestamp("2024-06-14"), "Euro 2024"),
    ("Copa América", pd.Timestamp("2024-06-20"), "Copa America 2024"),
    ("FIFA World Cup", pd.Timestamp("2018-06-14"), "2018 World Cup"),
    ("UEFA Euro", pd.Timestamp("2021-06-11"), "Euro 2020"),
]

rows = []
details = {}
for ttype, cutoff, label in TOURNAMENTS:
    past = r[r["date"] < cutoff]
    elo = compute_elo(past)
    att, defe, lavg = estimate_ratings(past, ref_date=cutoff)
    fut = r[(r["tournament"] == ttype) & (r["date"] >= cutoff)].dropna(subset=["home_score", "away_score"])
    if len(fut) == 0:
        continue
    probs_list, actual_list, recs = [], [], []
    for _, m in fut.iterrows():
        h, a = m["home_team"], m["away_team"]
        neutral = bool(m.get("neutral", False))
        elo_diff = elo.ratings.get(h, 1500.0) - elo.ratings.get(a, 1500.0)
        p = match_outcome_probs(h, a, att, defe, lavg, neutral=neutral, elo_diff=elo_diff)
        probs_list.append([p["ph"], p["pd"], p["pa"]])
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        if hs > as_:
            actual_list.append([1.0, 0.0, 0.0])
            actual = "H"
        elif hs < as_:
            actual_list.append([0.0, 0.0, 1.0])
            actual = "A"
        else:
            actual_list.append([0.0, 1.0, 0.0])
            actual = "D"
        probs = np.array([p["ph"], p["pd"], p["pa"]])
        pred = "H" if probs.argmax() == 0 else ("D" if probs.argmax() == 1 else "A")
        recs.append({
            "match": f"{h} vs {a}",
            "score": f"{hs}-{as_}",
            "ph": p["ph"], "pd": p["pd"], "pa": p["pa"],
            "pred": pred, "actual": actual, "correct": pred == actual,
        })
    probs_arr = np.array(probs_list)
    actual_arr = np.array(actual_list)
    rows.append({
        "tournament": label,
        "n_matches": len(fut),
        "brier": brier(probs_arr, actual_arr),
        "log_loss": log_loss(probs_arr, actual_arr),
        "rps": rps(probs_arr, actual_arr),
        "outcome_accuracy": sum(r["correct"] for r in recs) / len(recs),
    })
    details[label] = recs

df = pd.DataFrame(rows)
df.to_csv(OUT / "backtest_summary.csv", index=False)
with open(OUT / "backtest_summary.json", "w") as f:
    json.dump({"summary": rows}, f, indent=2)

print("Backtest summary:")
print(df.to_string(index=False))
print(f"\nSaved {OUT / 'backtest_summary.csv'}")
