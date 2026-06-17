"""Module: tournament simulator with the OFFICIAL 2026 FIFA R32 bracket.

Format (48 teams, 12 groups A..L of 4):
  * Group stage: 72 matches (June 11-27, 2026)
  * Top 2 in each group advance (24 teams)
  * Plus 8 best 3rd-placed teams -> 32 teams in Round of 32
  * R32: 16 matches (June 28 - July 3, 2026)
  * R16: 8 matches (July 4-7)
  * QF: 4 matches (July 9-11)
  * SF: 2 matches (July 14-15)
  * 3rd place: 1 match (July 18)
  * Final: 1 match (July 19, Philadelphia)

R32 bracket (from official FIFA PDF, v17 10042026):

  M73 (28 Jun):  2A  v  2B
  M74 (29 Jun):  1F  v  2C
  M75 (29 Jun):  1E  v  3[ABCDF]
  M76 (30 Jun):  1A  v  3[CEFHI]
  M77 (30 Jun):  1C  v  2F
  M78 (30 Jun):  2E  v  2I
  M79 (1 Jul):   1G  v  3[AEHIJ]
  M80 (1 Jul):   1L  v  3[EHIJK]
  M81 (1 Jul):   1I  v  3[CDFGH]
  M82 (2 Jul):   1B  v  3[EFGIJ]
  M83 (2 Jul):   1D  v  3[BEFIJ]
  M84 (2 Jul):   2D  v  2G
  M85 (3 Jul):   1H  v  2J
  M86 (3 Jul):   1K  v  3[DEIJL]
  M87 (3 Jul):   1J  v  2H
  M88 (3 Jul):   2K  v  2L

Note: matches are M73..M88.  The 3rd-placed teams are referenced by which
groups they're from.  In our simulation, the 8 best 3rds fill those slots.
The bracketed groups indicate which set of 3rds can fill that slot.

R16: 8 matches (4-7 July)
  W73 v W74
  W75 v W77
  W76 v W78
  W79 v W81
  W80 v W82
  W83 v W85
  W84 v W86
  W87 v W88

(QF, SF, Final as standard single-elim)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd


# Official FIFA 2026 R32 bracket.
# Each entry: (slot_a_label, slot_b_label, allowed_3rd_groups_for_a, allowed_3rd_groups_for_b)
# slot_a_label = '1X' = winner of group X, '2X' = runner-up, '3' = 3rd-placed team
# allowed_3rd_groups: list of group letters whose 3rd-placed teams are eligible for that slot
R32_BRACKET: list[tuple[str, str, list[str], list[str]]] = [
    # (slot_a, slot_b, allowed_a_if_3rd, allowed_b_if_3rd)
    ("2A",  "2B",  [], []),                          # M73
    ("1F",  "2C",  [], []),                          # M74
    ("1E",  "3",   [], ["A","B","C","D","F"]),       # M75
    ("1A",  "3",   [], ["C","E","F","H","I"]),       # M76
    ("1C",  "2F",  [], []),                          # M77
    ("2E",  "2I",  [], []),                          # M78
    ("1G",  "3",   [], ["A","E","H","I","J"]),       # M79
    ("1L",  "3",   [], ["E","H","I","J","K"]),       # M80
    ("1I",  "3",   [], ["C","D","F","G","H"]),       # M81
    ("1B",  "3",   [], ["E","F","G","I","J"]),       # M82
    ("1D",  "3",   [], ["B","E","F","I","J"]),       # M83
    ("2D",  "2G",  [], []),                          # M84
    ("1H",  "2J",  [], []),                          # M85
    ("1K",  "3",   [], ["D","E","I","J","L"]),       # M86
    ("1J",  "2H",  [], []),                          # M87
    ("2K",  "2L",  [], []),                          # M88
]

# R16 pairings: W{match_num} vs W{match_num}
R16_PAIRS: list[tuple[int, int]] = [
    (73, 74), (75, 77), (76, 78), (79, 81),
    (80, 82), (83, 85), (84, 86), (87, 88),
]

# QF: W{r16_match} pairs (r16 matches numbered 1-8)
# From the official bracket, QF pairings:
QF_PAIRS: list[tuple[int, int]] = [
    (1, 2), (3, 4), (5, 6), (7, 8),
]
# SF: standard
SF_PAIRS: list[tuple[int, int]] = [
    (1, 2), (3, 4),
]


def identify_third_slot_matches() -> list[int]:
    """Return the R32 match numbers that have a 3rd-placed team in slot_b."""
    return [i + 73 for i, (_, _, _, b) in enumerate(R32_BRACKET) if b == ["3"] or (isinstance(b, list) and b)]


def precompute_pair_cache(teams: list[str], predict) -> dict:
    """For each ordered pair (a, b) with a != b, precompute the score matrix."""
    cache = {}
    for a in teams:
        for b in teams:
            if a == b:
                continue
            p = predict(a, b, knockout=False)
            matrix = p["matrix"].astype(np.float64)
            s = matrix.sum()
            if s > 0:
                matrix = matrix / s
            flat = matrix.ravel().astype(np.float64)
            cache[(a, b)] = {
                "flat": flat,
                "lam_a": float(p["lambda_home"]),
                "lam_b": float(p["lambda_away"]),
                "ph": float(p["ph"]),
                "pd": float(p["pd"]),
                "pa": float(p["pa"]),
            }
    return cache


def _sample_flat(flat, rng: np.random.Generator) -> int:
    return int(rng.choice(flat.size, p=flat))


def knockout_winner(a: str, b: str, cache: dict, rng: np.random.Generator) -> tuple[str, int, int, str]:
    """Simulate one knockout match."""
    entry = cache[(a, b)]
    idx = _sample_flat(entry["flat"], rng)
    gh, ga = idx // 9, idx % 9
    if gh == ga:
        if rng.random() < entry["lam_a"] / 40.0:
            gh += 1
        if gh == ga and rng.random() < entry["lam_b"] / 40.0:
            ga += 1
        if gh == ga:
            pen_w = entry["ph"] + 0.5 * entry["pd"]
            if rng.random() < pen_w:
                return a, gh, ga, "PEN"
            return b, gh, ga, "PEN"
    return (a if gh > ga else b), gh, ga, ("REG" if (gh + ga > 0 and gh != ga) else "REG")


def simulate_group(teams: list[str], cache: dict, rng: np.random.Generator) -> tuple[dict, list]:
    """Round-robin in a group. Returns (table_dict, matches_list)."""
    pts = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}
    matches = []
    for i, h in enumerate(teams):
        for a in teams[i + 1:]:
            entry = cache[(h, a)]
            idx = _sample_flat(entry["flat"], rng)
            gh, gga = idx // 9, idx % 9
            matches.append((h, a, gh, gga))
            if gh > gga:
                pts[h] += 3
            elif gh < gga:
                pts[a] += 3
            else:
                pts[h] += 1
                pts[a] += 1
            gf[h] += gh
            gf[a] += gga
            ga[h] += gga
            ga[a] += gh
    table = {t: (pts[t], gf[t] - ga[t], gf[t]) for t in teams}
    ranked = sorted(teams, key=lambda t: (-pts[t], -(gf[t] - ga[t]), -gf[t]))
    return table, ranked, matches


def _assign_thirds_to_slots(best_8: list, third_slot_indices: list[int]) -> dict:
    """Use bipartite matching to assign the 8 best 3rd-placed teams to the 8
    3rd-slots in R32, respecting each slot's allowed-group constraint.

    best_8: list of (group_letter, team, pts, gd, gf) sorted best-first
    third_slot_indices: list of R32 match indices (0-15) that have a 3rd-slot

    Returns dict mapping match_idx -> team_name. Raises if no valid assignment.
    """
    # Build edges: for each (slot_idx, team_idx) check eligibility
    edges = []  # (slot_idx_within_third_slots, team_idx_within_best_8)
    for i, mi in enumerate(third_slot_indices):
        la, lb, aa, ab = R32_BRACKET[mi]
        for j, (g, t, *_rest) in enumerate(best_8):
            if g in ab:
                edges.append((i, j))
    # Try NetworkX first
    try:
        import networkx as nx
        G = nx.Graph()
        # Pre-add all slot nodes and team nodes (avoids KeyError on add_edge)
        slot_nodes = [f"slot_{i}" for i in range(len(third_slot_indices))]
        team_nodes = [f"team_{j}" for j in range(len(best_8))]
        for n in slot_nodes:
            G.add_node(n, bipartite=0)
        for n in team_nodes:
            G.add_node(n, bipartite=1)
        for i, j in edges:
            G.add_edge(f"slot_{i}", f"team_{j}")
        matcher = nx.bipartite.maximum_matching(G, top_nodes=slot_nodes)
        result = {}
        for slot_node in slot_nodes:
            if slot_node in matcher:
                team_node = matcher[slot_node]
                j = int(team_node.split("_")[1])
                i = int(slot_node.split("_")[1])
                mi = third_slot_indices[i]
                result[mi] = best_8[j][1]
        if len(result) == len(third_slot_indices):
            return result
        # Otherwise fall through to permutation
    except ImportError:
        pass

    # Fallback: try all permutations (8! = 40320)
    from itertools import permutations
    slots_info = []
    for i, mi in enumerate(third_slot_indices):
        la, lb, aa, ab = R32_BRACKET[mi]
        slots_info.append((i, ab))
    for perm in permutations(range(len(best_8))):
        ok = True
        result = {}
        for (i, allowed), j in zip(slots_info, perm):
            g, t, *_ = best_8[j]
            if g not in allowed:
                ok = False
                break
            mi = third_slot_indices[i]
            result[mi] = t
        if ok:
            return result
    raise RuntimeError("No valid 3rd-placed assignment found")


def simulate_tournament(groups: dict[str, list[str]], cache: dict, rng: np.random.Generator) -> dict:
    """Simulate a complete tournament using the OFFICIAL FIFA 2026 bracket."""
    # Group stage
    group_tables = {}
    group_ranked = {}
    for letter, tlist in groups.items():
        table, ranked, matches = simulate_group(tlist, cache, rng)
        group_tables[letter] = {t: {"Pts": table[t][0], "GD": table[t][1], "GF": table[t][2]} for t in tlist}
        group_ranked[letter] = ranked

    # Best 8 3rds
    third_info = []
    for letter, ranked in group_ranked.items():
        t = ranked[2]
        info = group_tables[letter][t]
        third_info.append((letter, t, info["Pts"], info["GD"], info["GF"]))
    third_info.sort(key=lambda r: (-r[2], -r[3], -r[4]))
    best_8 = third_info[:8]  # list of (group, team, pts, gd, gf)

    # Find all R32 match indices that have a 3rd-slot
    third_slot_indices = [
        mi for mi, (la, lb, aa, ab) in enumerate(R32_BRACKET)
        if lb == "3" or la == "3"
    ]
    # Assign 3rds to slots via bipartite matching
    third_assignments = _assign_thirds_to_slots(best_8, third_slot_indices)

    # Build slot map
    slots = {}
    for letter, ranked in group_ranked.items():
        slots[f"1{letter}"] = ranked[0]
        slots[f"2{letter}"] = ranked[1]
    for mi, team in third_assignments.items():
        slots[f"3_{mi}"] = team  # 3rd-placed slot in R32 match mi

    # R32 matches
    r32_winners = {}
    r32_losers = {}
    r32_results = []
    for mi, (la, lb, aa, ab) in enumerate(R32_BRACKET):
        match_num = 73 + mi
        if la == "3":
            a = third_assignments.get(mi)
        else:
            a = slots.get(la)
        if lb == "3":
            b = third_assignments.get(mi)
        else:
            b = slots.get(lb)
        if a is None or b is None:
            # Incomplete bracket (shouldn't happen with 8 3rds qualifying)
            continue
        w, gh, ga, how = knockout_winner(a, b, cache, rng)
        r32_winners[match_num] = w
        r32_losers[match_num] = b if w == a else a
        r32_results.append({"match": match_num, "a": a, "b": b, "gh": gh, "ga": ga, "winner": w, "how": how})

    # R16
    r16_winners = {}
    r16_results = []
    for r16_idx, (m1, m2) in enumerate(R16_PAIRS, 1):
        a = r32_winners.get(m1)
        b = r32_winners.get(m2)
        if a is None or b is None:
            continue
        w, gh, ga, how = knockout_winner(a, b, cache, rng)
        r16_winners[r16_idx] = w
        r16_results.append({"r16": r16_idx, "a": a, "b": b, "gh": gh, "ga": ga, "winner": w, "how": how})

    # QF
    qf_winners = {}
    qf_results = []
    for qf_idx, (r1, r2) in enumerate(QF_PAIRS, 1):
        a = r16_winners.get(r1)
        b = r16_winners.get(r2)
        if a is None or b is None:
            continue
        w, gh, ga, how = knockout_winner(a, b, cache, rng)
        qf_winners[qf_idx] = w
        qf_results.append({"qf": qf_idx, "a": a, "b": b, "gh": gh, "ga": ga, "winner": w, "how": how})

    # SF
    sf_winners = {}
    sf_losers = {}
    sf_results = []
    for sf_idx, (q1, q2) in enumerate(SF_PAIRS, 1):
        a = qf_winners.get(q1)
        b = qf_winners.get(q2)
        if a is None or b is None:
            continue
        w, gh, ga, how = knockout_winner(a, b, cache, rng)
        sf_winners[sf_idx] = w
        sf_losers[sf_idx] = b if w == a else a
        sf_results.append({"sf": sf_idx, "a": a, "b": b, "gh": gh, "ga": ga, "winner": w, "how": how})

    # Final
    final = None
    if 1 in sf_winners and 2 in sf_winners:
        a, b = sf_winners[1], sf_winners[2]
        w, gh, ga, how = knockout_winner(a, b, cache, rng)
        final = {"a": a, "b": b, "gh": gh, "ga": ga, "winner": w, "how": how}

    # 3rd place match (Bronze final)
    third_place = None
    if 1 in sf_losers and 2 in sf_losers:
        a, b = sf_losers[1], sf_losers[2]
        w, gh, ga, how = knockout_winner(a, b, cache, rng)
        third_place = {"a": a, "b": b, "gh": gh, "ga": ga, "winner": w, "how": how}

    return {
        "group_ranked": group_ranked,
        "group_tables": group_tables,
        "third_qualifiers": [r[1] for r in best_8],
        "third_assignments": third_assignments,  # mi -> team
        "r32_results": r32_results,
        "r16_results": r16_results,
        "qf_results": qf_results,
        "sf_results": sf_results,
        "final": final,
        "third_place": third_place,
        "winner": final["winner"] if final else None,
        "runner_up": final["b"] if final and final["winner"] == final["a"] else (final["a"] if final else None),
    }


if __name__ == "__main__":
    import time
    teams = [f"T{i}" for i in range(1, 49)]
    groups = {chr(ord("A") + i): teams[i * 4:(i + 1) * 4] for i in range(12)}

    def predict(a, b, knockout=False):
        m = np.full((9, 9), 1.0 / 81.0)
        return {"ph": 0.4, "pd": 0.3, "pa": 0.3, "matrix": m, "lambda_home": 1.3, "lambda_away": 1.0}

    cache = precompute_pair_cache(teams, predict)
    t0 = time.time()
    for _ in range(2000):
        simulate_tournament(groups, cache, np.random.default_rng(0))
    t1 = time.time()
    print(f"2000 sims in {t1 - t0:.2f}s")
