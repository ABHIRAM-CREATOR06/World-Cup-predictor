"""Module: attack/defense ratings + Poisson goal model.

The idea is: for each team we estimate how often they score and concede,
relative to the league average. Then the expected goals in a match are

    lambda_home = base * attack_home * defense_away * home_advantage
    lambda_away = base * attack_away * defense_home

We estimate attack/defense from the same historical data we used for ELO,
weighted by recency (exponential decay) and tournament importance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

LEAGUE_AVG_GOALS = 1.35  # ~ long-run goals scored per team per game (full internationals)

# Correlation parameter for bivariate Poisson (positive correlation between teams' goal counts)
RHO = 0.15

# Recency half-life: matches older than 2*HALF_LIFE days have ~25% the weight
HALF_LIFE_DAYS = 730  # 2 years

# Tournament importance for weight in attack/defense estimation
WEIGHTS = {
    "Friendly": 1.0,
    "FIFA World Cup qualification": 1.3,
    "UEFA Euro qualification": 1.3,
    "African Cup of Nations qualification": 1.2,
    "AFC Asian Cup qualification": 1.2,
    "CONCACAF Championship qualification": 1.2,
    "Copa América qualification": 1.2,
    "UEFA Nations League": 1.4,
    "CONCACAF Nations League": 1.3,
    "African Cup of Nations": 1.4,
    "AFC Asian Cup": 1.4,
    "CONCACAF Gold Cup": 1.4,
    "Copa América": 1.4,
    "UEFA Euro": 1.5,
    "FIFA Confederations Cup": 1.4,
    "Olympic Games": 1.0,
    "FIFA World Cup": 1.6,
}

HOME_ADV_GOALS = 1.18  # home teams score ~1.18x away-equivalent in non-neutral games


def _tournament_weight(t: str) -> float:
    return WEIGHTS.get(t, 1.0)


def _recency_weight(when: pd.Timestamp, ref: pd.Timestamp) -> float:
    days = (ref - when).days
    if days < 0:
        return 1.0  # future
    return math.exp(-days * math.log(2) / HALF_LIFE_DAYS)


def estimate_ratings(
    results: pd.DataFrame,
    *,
    ref_date: pd.Timestamp | None = None,
    min_matches: int = 5,
    shrink_to: float = 1.0,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Return (attack, defense, league_avg) dicts.

    `attack[t]` is multiplicative: 1.0 = league-average scoring, 1.2 = 20%
    better than league average.
    `defense[t]` is multiplicative: 1.0 = league-average concession, 0.85 =
    conceding 15% less than average (a good defense).

    For teams with very few matches we shrink their estimate toward 1.0
    (the league average). Shrinkage is controlled by `shrink_to` (the prior
    multiplier; usually 1.0) and the match-count-based weight.
    """
    if ref_date is None:
        ref_date = results["date"].max()

    df = results.dropna(subset=["home_score", "away_score"]).copy()
    df = df[df["date"] <= ref_date]
    df["w_recency"] = df["date"].apply(lambda d: _recency_weight(d, ref_date))
    df["w_tournament"] = df["tournament"].map(_tournament_weight).fillna(1.0)
    df["w"] = df["w_recency"] * df["w_tournament"]

    # League average
    total_w = df["w"].sum() * 2  # two teams per match
    league_avg = (df["home_score"] * df["w"] + df["away_score"] * df["w"]).sum() / total_w
    league_avg = float(league_avg) if league_avg > 0 else LEAGUE_AVG_GOALS

    # Per-team scored and conceded (weighted)
    teams = set(df["home_team"]) | set(df["away_team"])
    attack_raw: dict[str, float] = {}
    defense_raw: dict[str, float] = {}
    n_matches: dict[str, float] = {}

    for t in teams:
        # home + away split
        home = df[df["home_team"] == t]
        away = df[df["away_team"] == t]
        scored = (home["home_score"] * home["w"]).sum() + (away["away_score"] * away["w"]).sum()
        conceded = (home["away_score"] * home["w"]).sum() + (away["home_score"] * away["w"]).sum()
        w_sum = (home["w"].sum() + away["w"].sum())
        if w_sum == 0:
            continue
        attack_raw[t] = scored / (w_sum * league_avg)
        defense_raw[t] = conceded / (w_sum * league_avg)
        n_matches[t] = w_sum

    # Shrink toward 1.0 for low-sample teams
    attack = {}
    defense = {}
    for t in teams:
        n = n_matches.get(t, 0.0)
        # Effective number of games (each weight is roughly 1.0)
        s = min(1.0, n / max(min_matches, 1))
        attack[t] = s * attack_raw.get(t, 1.0) + (1 - s) * shrink_to
        defense[t] = s * defense_raw.get(t, 1.0) + (1 - s) * shrink_to

    return attack, defense, league_avg


def expected_goals(
    home: str,
    away: str,
    attack: dict[str, float],
    defense: dict[str, float],
    league_avg: float,
    *,
    neutral: bool = False,
    elo_diff: float | None = None,
    elo_scale: float = 0.06,
) -> tuple[float, float]:
    """Return (lambda_home, lambda_away) for a Poisson model.

    If `elo_diff` is provided (home_elo - away_elo), use it to nudge the
    expectation. We map 100 ELO ≈ `elo_scale` extra goals per team, capped.
    """
    att_h = attack.get(home, 1.0)
    def_a = defense.get(away, 1.0)
    att_a = attack.get(away, 1.0)
    def_h = defense.get(home, 1.0)

    lam_h = league_avg * att_h * def_a
    lam_a = league_avg * att_a * def_h

    if not neutral:
        lam_h *= HOME_ADV_GOALS
        lam_a /= HOME_ADV_GOALS

    if elo_diff is not None:
        # positive elo_diff -> more home goals, fewer away
        nudge = math.tanh(elo_diff / 600.0) * elo_scale
        lam_h *= (1.0 + nudge)
        lam_a *= (1.0 - nudge)

    # Cap to avoid degenerate Poisson tails
    lam_h = max(0.15, min(lam_h, 5.5))
    lam_a = max(0.15, min(lam_a, 5.5))
    return lam_h, lam_a


def match_outcome_probs(
    home: str,
    away: str,
    attack: dict[str, float],
    defense: dict[str, float],
    league_avg: float,
    *,
    neutral: bool = False,
    elo_diff: float | None = None,
    max_goals: int = 8,
    rho: float = RHO,
) -> dict:
    """Compute (p_home, p_draw, p_away) and full score matrix up to max_goals.

    Returns a dict with keys: 'ph', 'pd', 'pa', 'matrix' (numpy (max_goals+1) x (max_goals+1))
    """
    lam_h, lam_a = expected_goals(
        home, away, attack, defense, league_avg, neutral=neutral, elo_diff=elo_diff
    )
    gh = poisson.pmf(np.arange(max_goals + 1), lam_h)
    ga = poisson.pmf(np.arange(max_goals + 1), lam_a)
    matrix = np.outer(gh, ga)

    if rho != 0:
        rho_mat = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if i > 0 and j > 0:
                    rho_mat[i, j] = min(gh[i] * ga[j], rho)
        matrix = np.maximum(0, matrix + rho_mat)
        matrix = matrix / matrix.sum()

    ph = float(np.tril(matrix, -1).sum())
    pd = float(np.diag(matrix).sum())
    pa = float(np.triu(matrix, 1).sum())
    return {
        "ph": ph, "pd": pd, "pa": pa,
        "matrix": matrix,
        "lambda_home": lam_h, "lambda_away": lam_a,
    }


if __name__ == "__main__":
    from data import load_results, load_former_names, build_name_map, normalize_teams

    r = load_results()
    fn = load_former_names()
    nm = build_name_map(fn)
    r = normalize_teams(r, nm, fn)

    ref = pd.Timestamp("2022-11-20")  # day before 2022 WC
    att, defe, lavg = estimate_ratings(r, ref_date=ref)

    # Demo: 2022 WC opener Qatar vs Ecuador
    print(f"league avg: {lavg:.3f}")
    p = match_outcome_probs("Qatar", "Ecuador", att, defe, lavg, neutral=True)
    print(f"Qatar vs Ecuador 2022:  P(H)={p['ph']:.3f}  P(D)={p['pd']:.3f}  P(A)={p['pa']:.3f}")
    print(f"  lambdas: {p['lambda_home']:.2f}, {p['lambda_away']:.2f}")
    print(f"  most likely score: ", end="")
    idx = np.unravel_index(p["matrix"].argmax(), p["matrix"].shape)
    print(f"{idx[0]}-{idx[1]}  prob={p['matrix'][idx]:.3f}")
    print(f"  top 5 scorelines: ", end="")
    flat = [(i, j, p["matrix"][i, j]) for i in range(9) for j in range(9)]
    flat.sort(key=lambda x: -x[2])
    for i, j, prob in flat[:5]:
        print(f"{i}-{j}({prob:.3f})", end=" ")
    print()
