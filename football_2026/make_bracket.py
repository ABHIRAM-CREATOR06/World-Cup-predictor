"""Generate a visual bracket of the most likely 2026 WC knockout path
using the OFFICIAL FIFA 2026 bracket structure."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\abhir\.mavis\sessions\mvs_fbdb1a4d0ee34fc9b22f88407249a52a\workspace\football_2026")

from data import load_results, load_former_names, build_name_map, normalize_teams, identify_2026_groups
from main_with_actuals import build_predictor, CUTOFF
from simulator import precompute_pair_cache, R32_BRACKET, R16_PAIRS, _assign_thirds_to_slots
import pandas as pd

OUT = Path(r"C:\Users\abhir\.mavis\sessions\mvs_fbdb1a4d0ee34fc9b22f88407249a52a\workspace\football_2026\out")

# Use the dataset WITH actuals folded in
r = load_results()
fn = load_former_names()
nm = build_name_map(fn)
r = normalize_teams(r, nm, fn)
actuals = pd.read_csv(OUT / "actual_results_2026.csv", parse_dates=["date"])
# Drop NA-score 2026 rows that we now have actuals for
actuals_keys = set(zip(actuals["date"].astype(str), actuals["home_team"], actuals["away_team"]))
mask_to_drop = (r["tournament"] == "FIFA World Cup") & r["date"].dt.year.eq(2026) & r["home_score"].isna() & (
    pd.Series(list(zip(r["date"].astype(str), r["home_team"], r["away_team"]))).isin(actuals_keys)
)
r = pd.concat([r[~mask_to_drop], actuals], ignore_index=True).sort_values("date").reset_index(drop=True)
r = r.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last").reset_index(drop=True)
groups = identify_2026_groups(r)
teams = sorted({t for g in groups.values() for t in g})
predict, elo, att, defe, lavg = build_predictor(r, CUTOFF)

OUT = Path(r"C:\Users\abhir\.mavis\sessions\mvs_fbdb1a4d0ee34fc9b22f88407249a52a\workspace\football_2026\out")

cache = precompute_pair_cache(teams, predict)

def knockout_winner(a, b):
    p = predict(a, b, knockout=True)
    if p["ph"] > p["pa"]:
        return a
    return b

# Most-likely group standings: use ACTUAL played matches + predict the rest
def simulate_group_deterministic(letter, teams_in_group):
    pts = {t: 0 for t in teams_in_group}
    gf = {t: 0 for t in teams_in_group}
    ga = {t: 0 for t in teams_in_group}
    # Use actual played matches first
    played_in = r[(r["tournament"] == "FIFA World Cup") & (r["date"].dt.year == 2026) &
                  r["home_score"].notna() &
                  r["home_team"].isin(teams_in_group)]
    for _, m in played_in.iterrows():
        h, a = m["home_team"], m["away_team"]
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        if h not in teams_in_group or a not in teams_in_group: continue
        if hs > as_: pts[h] += 3
        elif hs < as_: pts[a] += 3
        else: pts[h] += 1; pts[a] += 1
        gf[h] += hs; gf[a] += as_; ga[h] += as_; ga[a] += hs
    # Predict unplayed matches
    played_pairs = set(zip(played_in["home_team"], played_in["away_team"]))
    for i, h in enumerate(teams_in_group):
        for a in teams_in_group[i+1:]:
            if (h, a) in played_pairs or (a, h) in played_pairs:
                continue
            entry = cache[(h, a)]
            lam_h, lam_a = entry["lam_a"], entry["lam_b"]
            gh = max(0, round(lam_h))
            ga_ = max(0, round(lam_a))
            gf[h] += gh; gf[a] += ga_; ga[h] += ga_; ga[a] += gh
            if gh > ga_:
                pts[h] += 3
            elif gh < ga_:
                pts[a] += 3
            else:
                pts[h] += 1; pts[a] += 1
    return sorted(teams_in_group, key=lambda t: (-pts[t], -(gf[t]-ga[t]), -gf[t]))

group_ranked = {letter: simulate_group_deterministic(letter, g) for letter, g in groups.items()}

# Best 8 thirds: use actual played + predicted unplayed (same logic as sim_group)
played_in_all = r[(r["tournament"] == "FIFA World Cup") & (r["date"].dt.year == 2026) & r["home_score"].notna()]
played_pairs_all = set(zip(played_in_all["home_team"], played_in_all["away_team"]))
third_info = []
for letter, ranked in group_ranked.items():
    t = ranked[2]
    pts = 0; gd = 0; gf_ = 0
    # Real played matches involving this team
    for _, m in played_in_all.iterrows():
        h, a = m["home_team"], m["away_team"]
        if h == t or a == t:
            hs, as_ = int(m["home_score"]), int(m["away_score"])
            if hs > as_: pts += 3 if h == t else 0
            elif hs < as_: pts += 3 if a == t else 0
            else: pts += 1
            if h == t: gf_ += hs; gd += hs - as_
            else: gf_ += as_; gd += as_ - hs
    # Predicted unplayed matches involving this team
    for i, h in enumerate(groups[letter]):
        for a in groups[letter][i+1:]:
            if (h, a) in played_pairs_all or (a, h) in played_pairs_all:
                continue
            entry = cache[(h, a)]
            lam_h, lam_a = entry["lam_a"], entry["lam_b"]
            gh = max(0, round(lam_h))
            ga_ = max(0, round(lam_a))
            if h == t:
                pts += 3 if gh > ga_ else (1 if gh == ga_ else 0); gf_ += gh; gd += gh - ga_
            elif a == t:
                pts += 3 if ga_ > gh else (1 if gh == ga_ else 0); gf_ += ga_; gd += ga_ - gh
    third_info.append((letter, t, pts, gd, gf_))
third_info.sort(key=lambda r: (-r[2], -r[3], -r[4]))
best_8 = third_info[:8]

# Build R32 matches with 3rd-slots filled per FIFA rules
slots = {}
for letter, ranked in group_ranked.items():
    slots[f"1{letter}"] = ranked[0]
    slots[f"2{letter}"] = ranked[1]

r32 = []
# Use bipartite matching to assign 3rds to slots
third_slot_indices = [mi for mi, (la, lb, aa, ab) in enumerate(R32_BRACKET) if lb == "3" or la == "3"]
assignments = _assign_thirds_to_slots(best_8, third_slot_indices)
r32_dates = ["28 Jun", "28 Jun", "29 Jun", "29 Jun", "29 Jun", "30 Jun", "30 Jun", "30 Jun",
             "1 Jul", "1 Jul", "1 Jul", "2 Jul", "2 Jul", "2 Jul", "3 Jul", "3 Jul"]
for mi, (la, lb, aa, ab) in enumerate(R32_BRACKET):
    match_num = 73 + mi
    # Resolve slot a
    if la == "3":
        a = assignments.get(mi)
    else:
        a = slots.get(la)
    # Resolve slot b
    if lb == "3":
        b = assignments.get(mi)
    else:
        b = slots.get(lb)
    if a is None or b is None:
        continue
    w = knockout_winner(a, b)
    r32.append({"match_num": match_num, "slot_a": la, "slot_b": lb, "team_a": a, "team_b": b, "winner": w, "date": r32_dates[mi]})

# R16
r16 = []
for r16_idx, (m1, m2) in enumerate(R16_PAIRS, 1):
    a = next((r["winner"] for r in r32 if r["match_num"] == m1), None)
    b = next((r["winner"] for r in r32 if r["match_num"] == m2), None)
    if a and b:
        w = knockout_winner(a, b)
        r16.append({"r16_idx": r16_idx, "team_a": a, "team_b": b, "winner": w})

# QF
qf = []
for qf_idx, (r1, r2) in enumerate([(1,2),(3,4),(5,6),(7,8)], 1):
    a = next((r["winner"] for r in r16 if r["r16_idx"] == r1), None)
    b = next((r["winner"] for r in r16 if r["r16_idx"] == r2), None)
    if a and b:
        w = knockout_winner(a, b)
        qf.append({"qf_idx": qf_idx, "team_a": a, "team_b": b, "winner": w})

# SF
sf = []
for sf_idx, (q1, q2) in enumerate([(1,2),(3,4)], 1):
    a = next((r["winner"] for r in qf if r["qf_idx"] == q1), None)
    b = next((r["winner"] for r in qf if r["qf_idx"] == q2), None)
    if a and b:
        w = knockout_winner(a, b)
        sf.append({"sf_idx": sf_idx, "team_a": a, "team_b": b, "winner": w})

# Final
if len(sf) >= 2:
    a = sf[0]["winner"]; b = sf[1]["winner"]
    champion = knockout_winner(a, b)
    final = {"team_a": a, "team_b": b, "winner": champion}
else:
    final = None

# 3rd place match
third_place = None
if len(sf) >= 2:
    a = sf[0]["team_b"] if sf[0]["winner"] == sf[0]["team_a"] else sf[0]["team_a"]
    b = sf[1]["team_b"] if sf[1]["winner"] == sf[1]["team_a"] else sf[1]["team_a"]
    third_place = {"team_a": a, "team_b": b, "winner": knockout_winner(a, b)}

# Print
print("=" * 80)
print("MOST LIKELY 2026 WORLD CUP BRACKET (deterministic, official FIFA structure)")
print("=" * 80)
print()
print("GROUP WINNERS / RUNNERS-UP / 3RD-PLACED (8 best qualify):")
for letter in "ABCDEFGHIJKL":
    rk = group_ranked[letter]
    qual = "✓" if any(letter == b[0] for b in best_8) else " "
    print(f"  {letter}: 1st {rk[0]:20s}  2nd {rk[1]:20s}  3rd {rk[2]:20s} {qual}")

print()
print("ROUND OF 32 (28 Jun - 3 Jul):")
for r in r32:
    print(f"  M{r['match_num']:2d} ({r['date']:7s}) {r['slot_a']:4s} {r['team_a']:20s} vs {r['slot_b']:4s} {r['team_b']:20s} → {r['winner']}")

print()
print("ROUND OF 16 (4-7 Jul):")
for r in r16:
    print(f"  R16-{r['r16_idx']:2d}  {r['team_a']:25s} vs {r['team_b']:25s} → {r['winner']}")

print()
print("QUARTERFINALS (9-11 Jul):")
for r in qf:
    print(f"  QF-{r['qf_idx']:2d}  {r['team_a']:25s} vs {r['team_b']:25s} → {r['winner']}")

print()
print("SEMIFINALS (14-15 Jul):")
for r in sf:
    print(f"  SF-{r['sf_idx']:2d}  {r['team_a']:25s} vs {r['team_b']:25s} → {r['winner']}")

print()
print("3RD PLACE (18 Jul):")
if third_place:
    print(f"  3rd  {third_place['team_a']:25s} vs {third_place['team_b']:25s} → {third_place['winner']}")

print()
if final:
    print(f"FINAL (19 Jul, Philadelphia):  {final['team_a']:25s} vs {final['team_b']:25s} → 🏆 {final['winner']}")

# Save JSON
out = {
    "groups": {l: groups[l] for l in "ABCDEFGHIJKL"},
    "most_likely_group_standings": {l: group_ranked[l] for l in "ABCDEFGHIJKL"},
    "best_8_thirds": [(g, t, int(p), int(gd), int(gf)) for g, t, p, gd, gf in best_8],
    "R32": r32, "R16": r16, "QF": qf, "SF": sf, "final": final, "third_place": third_place,
}
with open(OUT / "most_likely_bracket.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved {OUT / 'most_likely_bracket.json'}")
