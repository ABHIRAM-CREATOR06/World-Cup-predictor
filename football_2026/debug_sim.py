"""Debug: trace through a single tournament sim."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\abhir\.mavis\sessions\mvs_fbdb1a4d0ee34fc9b22f88407249a52a\workspace\football_2026")

from data import load_results, load_former_names, build_name_map, normalize_teams, identify_2026_groups
from main import build_predictor, precompute_pair_cache, build_group_pairs, R32_PAIRS

r = load_results()
fn = load_former_names()
nm = build_name_map(fn)
r = normalize_teams(r, nm, fn)
groups = identify_2026_groups(r)
teams = sorted({t for g in groups.values() for t in g})

predict, elo, att, defe, lavg = build_predictor(r, pd.Timestamp("2026-06-10"))
cache = precompute_pair_cache(teams, predict)
group_pairs = build_group_pairs(groups)
r32_pairs = R32_PAIRS
rng = np.random.default_rng(42)

# One sim
def play(teams_round):
    winners = []
    for i in range(0, len(teams_round) - 1, 2):
        a, b = teams_round[i], teams_round[i + 1]
        entry = cache[(a, b)]
        idx = int(rng.choice(entry["flat"].size, p=entry["flat"]))
        gh, ga = idx // 9, idx % 9
        if gh == ga:
            if rng.random() < entry["lam_a"] / 40.0:
                gh += 1
            if gh == ga and rng.random() < entry["lam_b"] / 40.0:
                ga += 1
            if gh == ga:
                pen_w = entry["ph"] + 0.5 * entry["pd"]
                if rng.random() < pen_w:
                    winners.append(a)
                else:
                    winners.append(b)
                continue
        winners.append(a if gh > ga else b)
    return winners


# Group stage
group_results = {g: {} for g in groups}
for letter, a, b in group_pairs:
    entry = cache[(a, b)]
    idx = int(rng.choice(entry["flat"].size, p=entry["flat"]))
    gh, ga = idx // 9, idx % 9
    group_results[letter][(a, b)] = (gh, ga)

# Standings
group_ranked = {}
for letter, tlist in groups.items():
    pts = {t: 0 for t in tlist}
    gf = {t: 0 for t in tlist}
    ga = {t: 0 for t in tlist}
    for (a, b), (gh, gga) in group_results[letter].items():
        pts[a] += 3 if gh > gga else (1 if gh == gga else 0)
        pts[b] += 3 if gga > gh else (1 if gh == gga else 0)
        gf[a] += gh
        gf[b] += gga
        ga[a] += gga
        ga[b] += gh
    ranked = sorted(tlist, key=lambda t: (-pts[t], -(gf[t] - ga[t]), -gf[t]))
    group_ranked[letter] = ranked
    print(f"Group {letter}: {[(t, pts[t], gf[t]-ga[t], gf[t]) for t in ranked]}")

# Best 8 thirds
third_info = []
for letter, ranked in group_ranked.items():
    t = ranked[2]
    p_ = 0
    gd = 0
    gf_ = 0
    for (a, b), (gh, gga) in group_results[letter].items():
        if a == t:
            p_ += 3 if gh > gga else (1 if gh == gga else 0)
            gf_ += gh
            gd += gh - gga
        elif b == t:
            p_ += 3 if gga > gh else (1 if gh == gga else 0)
            gf_ += gga
            gd += gga - gh
    third_info.append((letter, t, p_, gd, gf_))
third_info.sort(key=lambda r: (-r[2], -r[3], -r[4]))
best_8 = third_info[:8]
print(f"\nBest 8 thirds: {best_8}")
elim = third_info[8:]
print(f"Eliminated: {elim}")

# Slots
slots = {}
for letter, ranked in group_ranked.items():
    slots[f"1{letter}"] = ranked[0]
    slots[f"2{letter}"] = ranked[1]
best_8_letters = [r[0] for r in best_8]
best_8_teams = dict(zip(best_8_letters, [r[1] for r in best_8]))
explicit = ["A", "B", "C", "D"]
in_explicit = [g for g in explicit if g in best_8_letters]
for g in in_explicit:
    slots[f"3{g}"] = best_8_teams[g]
remaining = [g for g in best_8_letters if g not in explicit]
for i, slot in enumerate(["E", "F", "G", "H"]):
    if i < len(remaining):
        slots[f"3{slot}"] = best_8_teams[remaining[i]]
    else:
        slots[f"3{slot}"] = None
print(f"\nSlots:")
for s, t in slots.items():
    print(f"  {s}: {t}")

# R32
r32_winners = []
for label_a, label_b in r32_pairs:
    a = slots.get(label_a)
    b = slots.get(label_b)
    if a is None or b is None:
        print(f"  R32 skip: {label_a}={a}, {label_b}={b}")
        continue
    w = play([a, b])
    if not w:
        print(f"  R32 play returned empty for {a} vs {b}!")
        continue
    r32_winners.append(w[0])

print(f"\nR32 winners ({len(r32_winners)}): {r32_winners}")

r16 = play(r32_winners)
print(f"R16 winners ({len(r16)}): {r16}")
qf = play(r16)
print(f"QF winners ({len(qf)}): {qf}")
sf = play(qf)
print(f"SF winners ({len(sf)}): {sf}")
fin = play(sf)
print(f"Final winner ({len(fin)}): {fin}")
