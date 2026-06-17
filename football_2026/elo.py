"""Module: team-level ELO ratings with goal-difference adjustment.

Implements a FiveThirtyEight-style ELO with:
  * home-advantage in expectation (delta ~ 100 ELO)
  * goal-difference multiplier (1, 1.5, 1.75, ..., then +0.5 per extra)
  * K-factor scaled by match importance (Friendly < WC < WC Final)
  * neutral-venue override of home advantage

Reference: https://www.eloratings.net/about
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

INITIAL_ELO = 1500.0

# Importance weight applied to K-factor
IMPORTANCE = {
    "Friendly": 1.0,
    "FIFA World Cup qualification": 1.4,
    "UEFA Euro qualification": 1.4,
    "African Cup of Nations qualification": 1.3,
    "AFC Asian Cup qualification": 1.3,
    "CONCACAF Championship qualification": 1.3,
    "Copa América qualification": 1.3,
    "UEFA Nations League": 1.4,
    "CONCACAF Nations League": 1.3,
    "African Cup of Nations": 1.5,
    "AFC Asian Cup": 1.5,
    "CONCACAF Gold Cup": 1.5,
    "Copa América": 1.5,
    "UEFA Euro": 1.6,
    "FIFA Confederations Cup": 1.5,
    "Olympic Games": 1.2,
    "FIFA World Cup": 1.7,  # group stage
}

K_BASE = 32.0
HOME_ADV = 95.0  # ELO points added to home team's rating for expectancy

# Goal-difference multiplier (FiveThirtyEight):
#   diff=1 -> 1.0
#   diff=2 -> 1.5
#   diff=3 -> 1.75
#   diff>=4 -> 1.75 + 0.1 per goal beyond 3
def _gd_mult(gd: int) -> float:
    gd = abs(gd)
    if gd == 0:
        return 1.0
    if gd == 1:
        return 1.0
    if gd == 2:
        return 1.5
    return 1.75 + 0.1 * (gd - 3)


def _k_factor(tournament: str, *, knockout: bool = False, final: bool = False) -> float:
    base = IMPORTANCE.get(tournament, 1.0) * K_BASE
    if knockout and tournament == "FIFA World Cup":
        base *= 1.10
    if final:
        base *= 1.15
    return base


def _win_expectancy(rating: float, opp_rating: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opp_rating - rating) / 400.0))


@dataclass
class EloResult:
    """Per-team final ratings and history."""
    ratings: dict[str, float]
    history: pd.DataFrame  # one row per match: team, opp, rating_before, rating_after, exp, actual


def compute_elo(
    results: pd.DataFrame,
    initial: float = INITIAL_ELO,
    include_unplayed: bool = False,
) -> EloResult:
    """Walk matches chronologically and update each team's ELO.

    Skips matches with missing scores. If include_unplayed=False, matches
    with NA scores (e.g. future fixtures) are ignored entirely. If True,
    the ELO is held fixed (no update) for those rows but they're kept in
    history with exp=NaN, which is useful for plotting.
    """
    df = results.sort_values("date").reset_index(drop=True).copy()
    df["_played"] = df["home_score"].notna() & df["away_score"].notna()

    ratings: dict[str, float] = {}
    rows: list[dict] = []

    for _, m in df.iterrows():
        h, a = m["home_team"], m["away_team"]
        h_r = ratings.get(h, initial)
        a_r = ratings.get(a, initial)

        # apply home advantage in expectation
        is_neutral = bool(m.get("neutral", False))
        h_exp = _win_expectancy(h_r + (0 if is_neutral else HOME_ADV), a_r)
        a_exp = 1.0 - h_exp

        row = {
            "date": m["date"],
            "home": h,
            "away": a,
            "tournament": m["tournament"],
            "neutral": is_neutral,
            "home_rating_before": h_r,
            "away_rating_before": a_r,
            "home_exp": h_exp,
            "away_exp": a_exp,
            "played": bool(m["_played"]),
        }

        if m["_played"]:
            hs, as_ = int(m["home_score"]), int(m["away_score"])
            actual_h = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
            gd = hs - as_
            mult = _gd_mult(gd)
            k = _k_factor(m["tournament"])

            new_h = h_r + k * mult * (actual_h - h_exp)
            new_a = a_r + k * mult * ((1 - actual_h) - a_exp)

            ratings[h] = new_h
            ratings[a] = new_a
            row.update(
                home_rating_after=new_h,
                away_rating_after=new_a,
                home_score=hs,
                away_score=as_,
                goal_diff_mult=mult,
                k=k,
            )
        else:
            row.update(
                home_rating_after=h_r,
                away_rating_after=a_r,
                home_score=np.nan,
                away_score=np.nan,
                goal_diff_mult=np.nan,
                k=np.nan,
            )

        rows.append(row)

    history = pd.DataFrame(rows)
    return EloResult(ratings=ratings, history=history)


def ratings_at(history: pd.DataFrame, when: pd.Timestamp) -> dict[str, float]:
    """Return the ELO rating for every team as of `when` (most recent update)."""
    df = history[history["date"] <= when]
    last_h = df.dropna(subset=["home_rating_after"]).groupby("home")["home_rating_after"].last()
    last_a = df.dropna(subset=["away_rating_after"]).groupby("away")["away_rating_after"].last()
    combined = pd.concat([last_h, last_a]).groupby(level=0).last().to_dict()
    # teams that appear only in unplayed fixtures (no rating_after) get skipped
    return combined


if __name__ == "__main__":
    from data import load_results, load_former_names, build_name_map, normalize_teams

    r = load_results()
    fn = load_former_names()
    nm = build_name_map(fn)
    r = normalize_teams(r, nm, fn)

    print("Computing ELO...")
    elo = compute_elo(r)
    print(f"Teams with rating: {len(elo.ratings)}")
    top = sorted(elo.ratings.items(), key=lambda kv: -kv[1])[:20]
    print("\nTop 20 ELO (last update):")
    for t, r_ in top:
        print(f"  {r_:7.1f}  {t}")

    # Sanity: ratings for the 2026 WC participants
    wc26 = r[(r["tournament"] == "FIFA World Cup") & (r["date"].dt.year == 2026)]
    wc_teams = sorted(set(wc26["home_team"]) | set(wc26["away_team"]))
    print(f"\n2026 WC teams ({len(wc_teams)}):")
    wc_ratings = sorted([(t, elo.ratings.get(t, INITIAL_ELO)) for t in wc_teams], key=lambda kv: -kv[1])
    for t, r_ in wc_ratings:
        print(f"  {r_:7.1f}  {t}")
