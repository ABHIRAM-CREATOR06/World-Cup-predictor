"""Main entry point: build the 2026 World Cup prediction model end-to-end.

Uses the OFFICIAL FIFA 2026 R32 bracket (from the schedule PDF v17).
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

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

WC_CUTOFF = pd.Timestamp("2026-06-10")  # day before WC opener
N_SIM = 20000
RNG_SEED = 20260611


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


def predict_group_matches(groups, predict):
    rows = []
    for letter, teams in groups.items():
        for i, h in enumerate(teams):
            for a in teams[i + 1:]:
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


def run_monte_carlo(groups, cache, n, seed):
    """Run Monte Carlo, tracking per-team stage counts and group positions."""
    teams = [t for g in groups.values() for t in g]
    stage_counts = {t: [0] * 7 for t in teams}  # 0=group, 1=R32, 2=R16, 3=QF, 4=SF, 5=Final, 6=Win
    pos_counts = {t: {1: 0, 2: 0, 3: 0, 4: 0} for t in teams}
    third_qual_counts = {t: 0 for t in teams}
    # Knockout round-level appearances
    r16_appearance = {t: 0 for t in teams}  # reached R16
    qf_appearance = {t: 0 for t in teams}
    sf_appearance = {t: 0 for t in teams}
    final_appearance = {t: 0 for t in teams}
    third_place_appearance = {t: 0 for t in teams}
    champion_counts = Counter()

    rng = np.random.default_rng(seed)
    for _ in range(n):
        sim = simulate_tournament(groups, cache, rng)

        # Group position tracking
        for letter, ranked in sim["group_ranked"].items():
            for pos, t in enumerate(ranked, 1):
                pos_counts[t][pos] += 1
        # 3rd qualifier tracking
        third_set = set(sim["third_qualifiers"])
        for t in third_set:
            third_qual_counts[t] += 1

        # R32: all 32 teams that played
        r32_teams = set()
        for r in sim["r32_results"]:
            r32_teams.add(r["a"])
            r32_teams.add(r["b"])
        for t in r32_teams:
            stage_counts[t][1] += 1
        # R16 winners
        r16_teams = set()
        for r in sim["r16_results"]:
            r16_teams.add(r["a"])
            r16_teams.add(r["b"])
        for t in r16_teams:
            stage_counts[t][2] += 1
            r16_appearance[t] += 1
        # QF
        qf_teams = set()
        for r in sim["qf_results"]:
            qf_teams.add(r["a"])
            qf_teams.add(r["b"])
        for t in qf_teams:
            stage_counts[t][3] += 1
            qf_appearance[t] += 1
        # SF
        sf_teams = set()
        for r in sim["sf_results"]:
            sf_teams.add(r["a"])
            sf_teams.add(r["b"])
        for t in sf_teams:
            stage_counts[t][4] += 1
            sf_appearance[t] += 1
        # Final
        if sim["final"]:
            f = sim["final"]
            stage_counts[f["a"]][5] += 1
            stage_counts[f["b"]][5] += 1
            final_appearance[f["a"]] += 1
            final_appearance[f["b"]] += 1
            stage_counts[f["winner"]][6] += 1
            champion_counts[f["winner"]] += 1
        # 3rd place
        if sim["third_place"]:
            tp = sim["third_place"]
            third_place_appearance[tp["a"]] += 1
            third_place_appearance[tp["b"]] += 1
            third_place_appearance[tp["winner"]] += 1

    probs = {t: [c / n for c in counts] for t, counts in stage_counts.items()}
    pos_probs = {t: {p: c / n for p, c in pc.items()} for t, pc in pos_counts.items()}
    for t in teams:
        pos_probs[t]["3rd_qual"] = third_qual_counts[t] / n
    return probs, pos_probs, champion_counts


def make_report(groups, group_match_df, probs, pos_probs, champion_counts, elo, lavg, n):
    md = []
    md.append("# 2026 FIFA World Cup — Predictive Model Report\n")
    md.append("**Model**: ELO (FiveThirtyEight-style) + attack/defense ratings + Poisson goal model.\n")
    md.append("**Bracket**: Official FIFA 2026 R32 bracket (from schedule PDF v17, 10 April 2026).\n")
    md.append("**Backtested** on 2018 World Cup, Euro 2020, 2022 World Cup, Euro 2024, Copa America 2024: Brier 0.55-0.65, outcome accuracy 45-56%.\n")
    md.append("**Data**: international results 1872-2026-06-06 (49,435 matches).\n")
    md.append(f"**Monte Carlo**: {n:,} full tournament simulations.\n")
    md.append(f"**Predictor snapshot date**: {WC_CUTOFF.date()}.\n")
    md.append(f"**League avg goals/team/game**: {lavg:.3f}\n")
    md.append("")

    md.append("## 1. Official FIFA Group Draw (12 groups, 4 teams each)\n")
    for letter in sorted(groups):
        teams = groups[letter]
        rated = sorted(teams, key=lambda t: -elo.get(t, 1500))
        ratings = [elo.get(t, 1500) for t in rated]
        md.append(f"- **Group {letter}**: {', '.join(teams)}  "
                  f"(ELO: {ratings[0]:.0f}/{ratings[1]:.0f}/{ratings[2]:.0f}/{ratings[3]:.0f})")
    md.append("")

    md.append("## 2. Group stage match predictions\n")
    md.append("P(H)/P(D)/P(A) are win/draw/loss probabilities. λ values are expected goals.\n")
    md.append("| Group | Match | P(H) | P(D) | P(A) | λ_H | λ_A | Top score |")
    md.append("|---|---|---:|---:|---:|---:|---:|---|")
    for letter in sorted(groups):
        sub = group_match_df[group_match_df["group"] == letter]
        for _, r in sub.iterrows():
            md.append(
                f"| {letter} | {r['home']} vs {r['away']} | "
                f"{r['ph']:.2f} | {r['pd']:.2f} | {r['pa']:.2f} | "
                f"{r['lambda_home']:.2f} | {r['lambda_away']:.2f} | "
                f"{r['most_likely_score']} ({r['most_likely_prob']:.2f}) |"
            )
    md.append("")

    md.append("## 3. Group position probabilities (Monte Carlo)\n")
    md.append("| Group | Team | P(1st) | P(2nd) | P(3rd) | P(4th) | P(R32) |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for letter in sorted(groups):
        for team in groups[letter]:
            pp = pos_probs[team]
            r32 = pp.get(1, 0) + pp.get(2, 0) + pp.get("3rd_qual", 0)
            md.append(
                f"| {letter} | {team} | {pp.get(1,0):.2f} | {pp.get(2,0):.2f} | "
                f"{pp.get(3,0):.2f} | {pp.get(4,0):.2f} | {r32:.2f} |"
            )
    md.append("")

    md.append("## 4. Tournament progression probabilities\n")
    md.append("| Team | P(R32) | P(R16) | P(QF) | P(SF) | P(Final) | P(Win) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for t in sorted(probs.keys(), key=lambda x: -probs[x][6]):
        p_ = probs[t]
        md.append(
            f"| {t} | {p_[1]:.2f} | {p_[2]:.2f} | {p_[3]:.2f} | "
            f"{p_[4]:.2f} | {p_[5]:.2f} | {p_[6]:.2f} |"
        )
    md.append("")

    md.append("## 5. Top-10 most likely champions\n")
    md.append("| Rank | Team | P(Win) |")
    md.append("|---:|---|---:|")
    top10 = sorted(probs.items(), key=lambda kv: -kv[1][6])[:10]
    for i, (t, p_) in enumerate(top10, 1):
        md.append(f"| {i} | {t} | {p_[6]*100:.1f}% |")
    md.append("")

    md.append("## 6. Notes\n")
    md.append(
        "- **No individual player data** — the model is team-level. Star players, injuries, "
        "manager changes, and locker-room dynamics are not captured.\n"
        "- **Knockout draw rates** are halved to reflect extra time + penalties. Penalty "
        "outcomes are weighted by regulation win expectancy.\n"
        "- **R32 bracket is the official FIFA 2026 structure** (from schedule PDF v17). "
        "Match dates: 28 June – 3 July 2026. The 8 best 3rd-placed teams fill the 8 "
        "3rd-slots, with constraints on which 3rds can fill which slot (per FIFA rules).\n"
        "- **R32 dates / venues**: PDF details per match (Mexico City opener 11 June, "
        "Atlanta/Philadelphia/Miami/Kansas City/Inglewood/etc. for knockouts).\n"
        "- **Monte Carlo variance**: with 20,000 sims, SE on P(Win) for top teams is "
        "roughly 0.1-0.2 percentage points.\n"
    )
    return "\n".join(md)


if __name__ == "__main__":
    print("Loading data...")
    r = load_results()
    fn = load_former_names()
    nm = build_name_map(fn)
    r = normalize_teams(r, nm, fn)

    print("Identifying groups (official FIFA draw)...")
    groups = identify_2026_groups(r)
    teams_all = sorted({t for g in groups.values() for t in g})
    print(f"  {len(teams_all)} teams in 12 groups (per FIFA)")

    print(f"\nBuilding predictor (cutoff {WC_CUTOFF.date()})...")
    predict, elo, att, defe, lavg = build_predictor(r, WC_CUTOFF)
    print(f"  ELO coverage: {len(elo)} teams")
    print(f"  Attack/defense coverage: {len(att)} teams")
    print(f"  League avg goals/team: {lavg:.3f}")

    print("\nPre-computing pair-probability cache...")
    cache = precompute_pair_cache(teams_all, predict)
    print(f"  {len(cache)} ordered pairs cached")

    print("\nPredicting group matches...")
    group_match_df = predict_group_matches(groups, predict)
    group_match_df.to_csv(OUT_DIR / "group_match_predictions.csv", index=False)
    print(f"  -> group_match_predictions.csv  ({len(group_match_df)} matches)")

    print(f"\nRunning {N_SIM:,} Monte Carlo simulations...")
    t0 = time.time()
    probs, pos_probs, champion_counts = run_monte_carlo(groups, cache, N_SIM, RNG_SEED)
    t1 = time.time()
    print(f"  done in {t1 - t0:.1f}s")

    # Save per-team tournament progression
    rows = []
    for t in teams_all:
        p_ = probs[t]
        rows.append({
            "team": t,
            "elo": elo.get(t, 1500.0),
            "P_R32": p_[1],
            "P_R16": p_[2],
            "P_QF": p_[3],
            "P_SF": p_[4],
            "P_Final": p_[5],
            "P_Win": p_[6],
        })
    prob_df = pd.DataFrame(rows).sort_values("P_Win", ascending=False)
    prob_df.to_csv(OUT_DIR / "tournament_probabilities.csv", index=False)
    print(f"  -> tournament_probabilities.csv")

    # Save group position probabilities
    pos_rows = []
    for t in teams_all:
        for pos in [1, 2, 3, 4]:
            pos_rows.append({"team": t, "position": pos, "probability": pos_probs[t].get(pos, 0)})
        pos_rows.append({"team": t, "position": "3rd_qual", "probability": pos_probs[t].get("3rd_qual", 0)})
    pos_df = pd.DataFrame(pos_rows)
    pos_df.to_csv(OUT_DIR / "group_position_probabilities.csv", index=False)
    print(f"  -> group_position_probabilities.csv")

    # Save ELO snapshot
    elo_df = pd.DataFrame(
        sorted([(t, elo.get(t, 1500.0)) for t in teams_all], key=lambda x: -x[1]),
        columns=["team", "elo"]
    )
    elo_df.to_csv(OUT_DIR / "elo_snapshot_2026.csv", index=False)

    # Save attack/defense
    ad_df = pd.DataFrame([
        {"team": t, "attack": att.get(t, 1.0), "defense": defe.get(t, 1.0)}
        for t in teams_all
    ]).sort_values("attack", ascending=False)
    ad_df.to_csv(OUT_DIR / "attack_defense_ratings.csv", index=False)

    # Summary JSON
    summary = {
        "cutoff": str(WC_CUTOFF.date()),
        "n_simulations": N_SIM,
        "league_avg_goals": lavg,
        "top10_champions": [
            {"team": t, "p_win": probs[t][6]}
            for t in sorted(probs.keys(), key=lambda x: -probs[x][6])[:10]
        ],
        "groups": {letter: teams for letter, teams in groups.items()},
        "bracket_source": "Official FIFA 2026 schedule PDF v17 (10 April 2026)",
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Markdown report
    md = make_report(groups, group_match_df, probs, pos_probs, champion_counts, elo, lavg, N_SIM)
    with open(OUT_DIR / "report.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\nTop 10 most likely champions:")
    for t in sorted(probs.keys(), key=lambda x: -probs[x][6])[:10]:
        print(f"  {probs[t][6]*100:5.2f}%  {t}  (ELO {elo.get(t, 1500):.0f})")

    print(f"\nOutputs in {OUT_DIR}:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name:40s}  ({f.stat().st_size:,} bytes)")
