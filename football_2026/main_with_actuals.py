"""End-to-end pipeline that:
  1. Loads results + actual 2026 WC results (MD1-2)
  2. Folds the 16 actual results into the dataset (overwriting NA-score 2026 rows)
  3. Recomputes ELO + attack/defense ratings as of TODAY (post-MD2)
  4. Predicts remaining group matches and runs 20k MC sims
  5. Backtests predictions vs actual results
  6. Regenerates all deliverables (HTML, PNG, MD, CSVs)
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\abhir\.mavis\sessions\mvs_fbdb1a4d0ee34fc9b22f88407249a52a\workspace\football_2026")

from data import (
    load_results, load_former_names, build_name_map, normalize_teams,
    identify_2026_groups,
)
from elo import compute_elo
from poisson import estimate_ratings, match_outcome_probs
from simulator import (
    precompute_pair_cache, simulate_tournament, R32_BRACKET, R16_PAIRS,
)

OUT_DIR = Path(r"C:\Users\abhir\.mavis\sessions\mvs_fbdb1a4d0ee34fc9b22f88407249a52a\workspace\football_2026\out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the post-MD2 cutoff (the user's "Today" is the latest data point)
CUTOFF = pd.Timestamp("2026-06-16")  # day after the user's "today" matches
N_SIM = 20000
RNG_SEED = 20260617


def build_predictor(results, ref_date):
    past = results[results["date"] < ref_date]
    elo_res = compute_elo(past)
    elo = elo_res.ratings
    att, defe, lavg = estimate_ratings(past, ref_date=ref_date)

    def predict(home: str, away: str, knockout: bool):
        elo_diff = elo.get(home, 1500.0) - elo.get(away, 1500.0)
        p = match_outcome_probs(home, away, att, defe, lavg, neutral=True, elo_diff=elo_diff)
        if knockout:
            new_pd = p["pd"] * 0.5
            scale = (1.0 - new_pd) / max(1e-9, p["ph"] + p["pa"])
            p["ph"] *= scale
            p["pa"] *= scale
            p["pd"] = new_pd
        return p

    return predict, elo, att, defe, lavg


def predict_group_matches(groups, predict, played_set=None):
    """Predict each unplayed group match. If played_set given, skip those."""
    played_set = played_set or set()
    rows = []
    for letter, teams in groups.items():
        for i, h in enumerate(teams):
            for a in teams[i + 1:]:
                key = (h, a)
                if key in played_set:
                    continue
                p = predict(h, a, False)
                idx = np.unravel_index(p["matrix"].argmax(), p["matrix"].shape)
                rows.append({
                    "group": letter,
                    "home": h,
                    "away": a,
                    "ph": p["ph"],
                    "pd": p["pd"],
                    "pa": p["pa"],
                    "lambda_home": p["lambda_home"],
                    "lambda_away": p["lambda_away"],
                    "most_likely_score": f"{idx[0]}-{idx[1]}",
                    "most_likely_prob": float(p["matrix"][idx]),
                })
    return pd.DataFrame(rows)


def backtest_actual(actuals_df, pre_cutoff_results, ref_date):
    """Score the model against the 16 actual results.

    For each match, we predict using the model state BEFORE the match,
    and compare to the actual outcome.
    """
    rows = []
    for _, m in actuals_df.iterrows():
        h, a = m["home_team"], m["away_team"]
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        # build predictor as of the day BEFORE this match
        match_date = m["date"] - pd.Timedelta(days=1)
        elo = compute_elo(pre_cutoff_results[pre_cutoff_results["date"] <= match_date]).ratings
        att, defe, lavg = estimate_ratings(
            pre_cutoff_results[pre_cutoff_results["date"] <= match_date],
            ref_date=match_date
        )
        elo_diff = elo.get(h, 1500.0) - elo.get(a, 1500.0)
        p = match_outcome_probs(h, a, att, defe, lavg, neutral=True, elo_diff=elo_diff)
        ph, pd_, pa = p["ph"], p["pd"], p["pa"]
        if hs > as_:
            actual = "H"
        elif hs < as_:
            actual = "A"
        else:
            actual = "D"
        pred = "H" if ph == max(ph, pd_, pa) else ("D" if pd_ == max(ph, pd_, pa) else "A")
        # Most likely score
        idx = np.unravel_index(p["matrix"].argmax(), p["matrix"].shape)
        top_score = f"{idx[0]}-{idx[1]}"
        rows.append({
            "date": m["date"].date(),
            "group": m.get("group", ""),
            "match": f"{h} vs {a}",
            "score": f"{hs}-{as_}",
            "ph": ph, "pd": pd_, "pa": pa,
            "pred_outcome": pred, "actual_outcome": actual,
            "correct_outcome": pred == actual,
            "lambda_h": p["lambda_home"], "lambda_a": p["lambda_away"],
            "top_score": top_score,
            "actual_top_match": top_score == f"{hs}-{as_}",
        })
    df = pd.DataFrame(rows)
    # Add Brier/log loss
    brier_list = []
    log_loss_list = []
    for _, r in df.iterrows():
        if r["actual_outcome"] == "H":
            target = [1, 0, 0]
        elif r["actual_outcome"] == "D":
            target = [0, 1, 0]
        else:
            target = [0, 0, 1]
        probs = [r["ph"], r["pd"], r["pa"]]
        brier_list.append(sum((p - t) ** 2 for p, t in zip(probs, target)))
        # clip for log
        eps = 1e-12
        log_loss_list.append(-np.log(max(probs[np.argmax(target)], eps)))
    df["brier"] = brier_list
    df["log_loss"] = log_loss_list
    return df


def run_monte_carlo(groups, cache, n, seed):
    teams = [t for g in groups.values() for t in g]
    stage_counts = {t: [0] * 7 for t in teams}
    pos_counts = {t: {1: 0, 2: 0, 3: 0, 4: 0} for t in teams}
    third_qual_counts = {t: 0 for t in teams}
    champion_counts = Counter()

    rng = np.random.default_rng(seed)
    for _ in range(n):
        sim = simulate_tournament(groups, cache, rng)
        for letter, ranked in sim["group_ranked"].items():
            for pos, t in enumerate(ranked, 1):
                pos_counts[t][pos] += 1
        for t in sim["third_qualifiers"]:
            third_qual_counts[t] += 1
        # R32
        r32_teams = set()
        for r in sim["r32_results"]:
            r32_teams.add(r["a"]); r32_teams.add(r["b"])
        for t in r32_teams: stage_counts[t][1] += 1
        # R16
        r16_teams = set()
        for r in sim["r16_results"]:
            r16_teams.add(r["a"]); r16_teams.add(r["b"])
        for t in r16_teams: stage_counts[t][2] += 1
        # QF
        qf_teams = set()
        for r in sim["qf_results"]:
            qf_teams.add(r["a"]); qf_teams.add(r["b"])
        for t in qf_teams: stage_counts[t][3] += 1
        # SF
        sf_teams = set()
        for r in sim["sf_results"]:
            sf_teams.add(r["a"]); sf_teams.add(r["b"])
        for t in sf_teams: stage_counts[t][4] += 1
        # Final
        if sim["final"]:
            f = sim["final"]
            stage_counts[f["a"]][5] += 1
            stage_counts[f["b"]][5] += 1
            stage_counts[f["winner"]][6] += 1
            champion_counts[f["winner"]] += 1

    probs = {t: [c / n for c in counts] for t, counts in stage_counts.items()}
    pos_probs = {t: {p: c / n for p, c in pc.items()} for t, pc in pos_counts.items()}
    for t in teams:
        pos_probs[t]["3rd_qual"] = third_qual_counts[t] / n
    return probs, pos_probs, champion_counts


if __name__ == "__main__":
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    r = load_results()
    fn = load_former_names()
    nm = build_name_map(fn)
    r = normalize_teams(r, nm, fn)

    # Load actual 2026 results
    actuals = pd.read_csv(OUT_DIR / "actual_results_2026.csv", parse_dates=["date"])
    print(f"Loaded {len(actuals)} actual 2026 results")

    # Replace NA-score 2026 rows with actual results
    r_before = r.copy()
    # Mark existing 2026 rows
    is_2026_na = (r["tournament"] == "FIFA World Cup") & (r["date"].dt.year == 2026) & r["home_score"].isna()
    print(f"Removing {is_2026_na.sum()} NA-score 2026 rows that we now have actuals for")

    # Drop the 2026 WC rows that we have actuals for
    # Match by (date, home_team, away_team)
    actuals_keys = set(zip(actuals["date"].astype(str), actuals["home_team"], actuals["away_team"]))
    mask_to_drop = (r["tournament"] == "FIFA World Cup") & r["date"].dt.year.eq(2026) & r["home_score"].isna() & (
        pd.Series(list(zip(r["date"].astype(str), r["home_team"], r["away_team"]))).isin(actuals_keys)
    )
    print(f"Matched actuals to drop: {mask_to_drop.sum()}")
    r = pd.concat([r[~mask_to_drop], actuals], ignore_index=True).sort_values("date").reset_index(drop=True)
    # Dedup
    r = r.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last").reset_index(drop=True)
    print(f"New dataset size: {len(r):,} matches (added {len(actuals)} actuals)")

    print()
    print("=" * 70)
    print(f"IDENTIFYING GROUPS (cutoff {CUTOFF.date()})")
    print("=" * 70)
    groups = identify_2026_groups(r)
    teams_all = sorted({t for g in groups.values() for t in g})
    print(f"{len(teams_all)} teams in 12 groups")

    # Determine which group matches have been played
    played = r[(r["tournament"] == "FIFA World Cup") & (r["date"].dt.year == 2026) & r["home_score"].notna()]
    played_pairs = set(zip(played["home_team"], played["away_team"]))
    print(f"Played WC matches: {len(played)}")

    print()
    print("=" * 70)
    print("BUILDING PREDICTOR (post-MD2)")
    print("=" * 70)
    predict, elo, att, defe, lavg = build_predictor(r, CUTOFF)
    print(f"ELO coverage: {len(elo)} teams, league avg: {lavg:.3f}")

    # Show updated ELO
    print("\nTop 15 ELO (post-MD2):")
    for t, e in sorted(elo.items(), key=lambda x: -x[1])[:15]:
        in_wc = t in teams_all
        marker = "  " if in_wc else "  "
        print(f"  {e:7.1f}  {t}")

    # Save updated ELO
    elo_df = pd.DataFrame(
        sorted([(t, e) for t, e in elo.items() if t in teams_all], key=lambda x: -x[1]),
        columns=["team", "elo"]
    )
    elo_df.to_csv(OUT_DIR / "elo_snapshot_2026_post_md2.csv", index=False)

    print()
    print("=" * 70)
    print("BACKTEST ON 16 ACTUAL MATCHES")
    print("=" * 70)
    # Build a pre-cutoff dataset (no actuals)
    pre_cutoff = pd.concat([r_before, ], ignore_index=True)  # just r_before (no actuals)
    bt = backtest_actual(actuals, r_before, CUTOFF)
    print(f"Outcome accuracy: {bt['correct_outcome'].mean()*100:.1f}%")
    print(f"Top-score accuracy: {bt['actual_top_match'].mean()*100:.1f}%")
    print(f"Mean Brier: {bt['brier'].mean():.3f}")
    print(f"Mean log-loss: {bt['log_loss'].mean():.3f}")
    print()
    print(bt[["date", "match", "score", "ph", "pd", "pa", "pred_outcome", "actual_outcome", "correct_outcome", "top_score"]].to_string(index=False))
    bt.to_csv(OUT_DIR / "backtest_md1_2.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'backtest_md1_2.csv'}")

    # Show actual group standings so far
    print()
    print("=" * 70)
    print("ACTUAL GROUP STANDINGS (after MD1-2)")
    print("=" * 70)
    for letter in "ABCDEFGHIJKL":
        teams_g = groups[letter]
        pts = {t: 0 for t in teams_g}
        gf = {t: 0 for t in teams_g}
        ga = {t: 0 for t in teams_g}
        played_in = played[played["home_team"].isin(teams_g) | played["away_team"].isin(teams_g)]
        for _, m in played_in.iterrows():
            h, a = m["home_team"], m["away_team"]
            if h not in teams_g: continue
            hs, as_ = int(m["home_score"]), int(m["away_score"])
            if hs > as_: pts[h] += 3
            elif hs < as_: pts[a] += 3
            else: pts[h] += 1; pts[a] += 1
            gf[h] += hs; gf[a] += as_; ga[h] += as_; ga[a] += hs
        if not any(pts.values()):
            print(f"  Group {letter}: no matches yet")
            continue
        ranked = sorted(teams_g, key=lambda t: (-pts[t], -(gf[t]-ga[t]), -gf[t]))
        print(f"  Group {letter}: " + " | ".join(f"{t} {pts[t]}-{gf[t]-ga[t]:+d}" for t in ranked))

    print()
    print("=" * 70)
    print("PREDICTING REMAINING GROUP MATCHES")
    print("=" * 70)
    group_match_df = predict_group_matches(groups, predict, played_set=played_pairs)
    group_match_df.to_csv(OUT_DIR / "group_match_predictions.csv", index=False)
    print(f"  -> group_match_predictions.csv ({len(group_match_df)} remaining matches)")

    print()
    print("=" * 70)
    print(f"MONTE CARLO ({N_SIM:,} sims)")
    print("=" * 70)
    cache = precompute_pair_cache(teams_all, predict)
    print(f"  {len(cache)} pair cache entries")
    t0 = time.time()
    probs, pos_probs, champion_counts = run_monte_carlo(groups, cache, N_SIM, RNG_SEED)
    print(f"  done in {time.time() - t0:.1f}s")

    # Save
    rows = []
    for t in teams_all:
        p_ = probs[t]
        rows.append({
            "team": t, "elo": elo.get(t, 1500.0),
            "P_R32": p_[1], "P_R16": p_[2], "P_QF": p_[3],
            "P_SF": p_[4], "P_Final": p_[5], "P_Win": p_[6],
        })
    prob_df = pd.DataFrame(rows).sort_values("P_Win", ascending=False)
    prob_df.to_csv(OUT_DIR / "tournament_probabilities.csv", index=False)

    pos_rows = []
    for t in teams_all:
        for pos in [1, 2, 3, 4]:
            pos_rows.append({"team": t, "position": pos, "probability": pos_probs[t].get(pos, 0)})
        pos_rows.append({"team": t, "position": "3rd_qual", "probability": pos_probs[t].get("3rd_qual", 0)})
    pos_df = pd.DataFrame(pos_rows)
    pos_df.to_csv(OUT_DIR / "group_position_probabilities.csv", index=False)

    # Attack/defense
    ad_df = pd.DataFrame([
        {"team": t, "attack": att.get(t, 1.0), "defense": defe.get(t, 1.0)}
        for t in teams_all
    ]).sort_values("attack", ascending=False)
    ad_df.to_csv(OUT_DIR / "attack_defense_ratings.csv", index=False)

    # Summary JSON
    summary = {
        "cutoff": str(CUTOFF.date()),
        "n_simulations": N_SIM,
        "league_avg_goals": lavg,
        "actuals_folded": len(actuals),
        "backtest_md1_2": {
            "outcome_accuracy": float(bt["correct_outcome"].mean()),
            "top_score_accuracy": float(bt["actual_top_match"].mean()),
            "brier": float(bt["brier"].mean()),
            "log_loss": float(bt["log_loss"].mean()),
        },
        "top10_champions": [
            {"team": t, "p_win": probs[t][6]}
            for t in sorted(probs.keys(), key=lambda x: -probs[x][6])[:10]
        ],
        "groups": {letter: teams for letter, teams in groups.items()},
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("=" * 70)
    print("TOP 10 CHAMPIONSHIP PROBABILITIES (post-MD2)")
    print("=" * 70)
    for t in sorted(probs.keys(), key=lambda x: -probs[x][6])[:10]:
        print(f"  {probs[t][6]*100:5.2f}%  {t:25s}  (ELO {elo.get(t, 1500):.0f})")

    print()
    print(f"Outputs in {OUT_DIR}:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name:40s}  ({f.stat().st_size:,} bytes)")
