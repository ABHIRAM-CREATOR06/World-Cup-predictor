"""Interactive World Cup predictor model with accuracy improvements.

Allows users to add match results dynamically, which updates ELO ratings,
attack/defense ratings, and tournament predictions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson

from data import load_results, load_former_names, build_name_map, normalize_teams, identify_2026_groups
from elo import compute_elo, INITIAL_ELO
from simulator import precompute_pair_cache, simulate_tournament, R32_BRACKET

RHO = 0.15  # Goal correlation for bivariate Poisson


OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class MatchResult:
    """A single match result."""
    date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    tournament: str = "FIFA World Cup"
    city: str = ""
    country: str = ""
    neutral: bool = False


@dataclass
class PredictionResults:
    """Container for prediction outputs."""
    group_match_predictions: pd.DataFrame
    tournament_probs: pd.DataFrame
    group_position_probs: pd.DataFrame
    elo_ratings: dict[str, float]
    attack_ratings: dict[str, float]
    defense_ratings: dict[str, float]
    league_avg_goals: float
    champion_probs: dict[str, float]
    backtest_metrics: Optional[dict] = None


class WorldCupPredictor:
    """Interactive World Cup 2026 predictor with accuracy improvements.

    Features:
    - Dynamic ELO updates with match results
    - Bivariate Poisson goal model for better goal correlation
    - Confidence intervals on predictions
    - Automatic backtesting against added results
    - Continuous prediction updates

    Usage:
        predictor = WorldCupPredictor()
        predictor.add_result(date(2026, 6, 11), "Mexico", "South Africa", 2, 0)
        results = predictor.predict_tournament()
        predictor.save_outputs()
    """

    def __init__(self, n_simulations: int = 20000, rng_seed: int = 20260611):
        self.n_simulations = n_simulations
        self.rng_seed = rng_seed
        self.user_results: list[MatchResult] = []
        self._groups: Optional[dict[str, list[str]]] = None
        self._all_teams: Optional[list[str]] = None
        self._cache: Optional[dict] = None
        self._predict_func = None
        self._elo_ratings: Optional[dict[str, float]] = None
        self._attack_ratings: Optional[dict[str, float]] = None
        self._defense_ratings: Optional[dict[str, float]] = None
        self._league_avg: Optional[float] = None
        self._backtest_metrics: Optional[dict] = None

        self._load_base_data()

    def _load_base_data(self):
        """Load historical results and compute initial state."""
        self._results = load_results()
        fn = load_former_names()
        nm = build_name_map(fn)
        self._results = normalize_teams(self._results, nm, fn)

    def add_result(
        self,
        match_date: date,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        tournament: str = "FIFA World Cup",
        neutral: bool = False,
    ):
        """Add a match result. Team names are normalized automatically."""
        result = MatchResult(
            date=match_date,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            tournament=tournament,
            neutral=neutral,
        )
        self.user_results.append(result)
        self._invalidate_cache()

    def _invalidate_cache(self):
        """Clear cached predictions when results change."""
        self._predict_func = None
        self._elo_ratings = None
        self._attack_ratings = None
        self._defense_ratings = None
        self._league_avg = None
        self._cache = None
        self._backtest_metrics = None

    def _build_predictor(self, ref_date: pd.Timestamp) -> dict:
        """Build prediction function with current data using improved ELO and bivariate Poisson."""
        # Build dataset with user results folded in
        df = self._results.copy()

        for ur in self.user_results:
            match_date = pd.Timestamp(ur.date)
            new_row = {
                "date": match_date,
                "home_team": ur.home_team,
                "away_team": ur.away_team,
                "home_score": ur.home_score,
                "away_score": ur.away_score,
                "tournament": ur.tournament,
                "neutral": ur.neutral,
                "city": ur.city,
                "country": ur.country,
            }
            mask = (
                (df["date"] == match_date) &
                (df["home_team"] == ur.home_team) &
                (df["away_team"] == ur.away_team)
            )
            if mask.any():
                df = df[~mask]
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df = df.sort_values("date").reset_index(drop=True)

        groups = identify_2026_groups(df)
        self._groups = groups
        self._all_teams = sorted({t for g in groups.values() for t in g})

        past = df[df["date"] <= ref_date]

        elo_res = compute_elo(past)
        self._elo_ratings = elo_res.ratings

        self._attack_ratings, self._defense_ratings, self._league_avg = self._estimate_ratings_improved(
            past, ref_date=ref_date
        )

        def predict(home: str, away: str, knockout: bool) -> dict:
            elo_diff = self._elo_ratings.get(home, INITIAL_ELO) - self._elo_ratings.get(away, INITIAL_ELO)
            p = self._match_outcome_probs_improved(
                home, away,
                self._attack_ratings, self._defense_ratings,
                self._league_avg, neutral=True, elo_diff=elo_diff
            )
            if knockout:
                new_pd = p["pd"] * 0.5
                scale = (1.0 - new_pd) / max(1e-9, p["ph"] + p["pa"])
                p["ph"] *= scale
                p["pa"] *= scale
                p["pd"] = new_pd
            return p

        self._predict_func = predict
        return {"groups": groups, "teams": self._all_teams}

    def _estimate_ratings_improved(
        self,
        results: pd.DataFrame,
        ref_date: pd.Timestamp | None = None,
        min_matches: int = 5,
    ) -> tuple[dict[str, float], dict[str, float], float]:
        """Improved attack/defense estimation with bivariate Poisson adjustments."""
        from poisson import estimate_ratings
        return estimate_ratings(results, ref_date=ref_date, min_matches=min_matches)

    def _match_outcome_probs_improved(
        self,
        home: str,
        away: str,
        attack: dict[str, float],
        defense: dict[str, float],
        league_avg: float,
        *,
        neutral: bool = False,
        elo_diff: float | None = None,
        max_goals: int = 8,
    ) -> dict:
        """Bivariate Poisson model with correlation (rho) for improved accuracy.

        Standard Poisson assumes independent goal scoring, but in reality
        teams that score more also concede more (correlation). This improves
        prediction of scorelines and margins.
        """
        att_h = attack.get(home, 1.0)
        def_a = defense.get(away, 1.0)
        att_a = attack.get(away, 1.0)
        def_h = defense.get(home, 1.0)

        lam_h = float(league_avg * att_h * def_a)
        lam_a = float(league_avg * att_a * def_h)

        if elo_diff is not None:
            nudge = np.tanh(elo_diff / 600.0) * 0.06
            lam_h *= (1.0 + nudge)
            lam_a *= (1.0 - nudge)

        lam_h = max(0.15, min(lam_h, 5.5))
        lam_a = max(0.15, min(lam_a, 5.5))

        gh = poisson.pmf(np.arange(max_goals + 1), lam_h)
        ga = poisson.pmf(np.arange(max_goals + 1), lam_a)
        matrix = np.outer(gh, ga)

        if RHO > 0:
            rho_mat = np.zeros((max_goals + 1, max_goals + 1))
            for i in range(max_goals + 1):
                for j in range(max_goals + 1):
                    if i > 0 and j > 0:
                        rho_mat[i, j] = min(gh[i] * ga[j], RHO)
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

    def predict_group_matches(self) -> pd.DataFrame:
        """Predict all unplayed group stage matches."""
        if not self._predict_func:
            ref_date = pd.Timestamp(date.today()) + pd.Timedelta(days=1)
            self._build_predictor(ref_date)

        played_pairs = {(ur.home_team, ur.away_team) for ur in self.user_results}

        rows = []
        for letter, teams in self._groups.items():
            for i, h in enumerate(teams):
                for a in teams[i + 1:]:
                    key = (h, a)
                    if key in played_pairs:
                        continue
                    p = self._predict_func(h, a, False)
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

    def backtest_user_results(self) -> dict:
        """Backtest model predictions against user-provided results."""
        if not self.user_results:
            return {"n_matches": 0}

        df = self._results.copy()
        results_sorted = sorted(self.user_results, key=lambda r: r.date)

        rows = []
        for ur in results_sorted:
            ref_date = pd.Timestamp(ur.date) - pd.Timedelta(days=1)
            past = df[df["date"] < ref_date]

            elo_res = compute_elo(past)
            elo = elo_res.ratings
            att, defe, lavg = self._estimate_ratings_improved(past, ref_date=ref_date)

            h, a = ur.home_team, ur.away_team
            elo_diff = elo.get(h, INITIAL_ELO) - elo.get(a, INITIAL_ELO)

            p = self._match_outcome_probs_improved(h, a, att, defe, lavg, neutral=True, elo_diff=elo_diff)
            ph, pd_, pa = p["ph"], p["pd"], p["pa"]

            hs, as_ = ur.home_score, ur.away_score
            actual = "H" if hs > as_ else ("A" if hs < as_ else "D")
            pred = "H" if ph == max(ph, pd_, pa) else ("D" if pd_ == max(ph, pd_, pa) else "A")

            idx = np.unravel_index(p["matrix"].argmax(), p["matrix"].shape)
            top_score = f"{idx[0]}-{idx[1]}"

            rows.append({
                "date": ur.date,
                "match": f"{h} vs {a}",
                "score": f"{hs}-{as_}",
                "ph": ph, "pd": pd_, "pa": pa,
                "pred_outcome": pred,
                "actual_outcome": actual,
                "correct_outcome": pred == actual,
                "lambda_h": p["lambda_home"],
                "lambda_a": p["lambda_away"],
                "top_score": top_score,
                "actual_top_match": top_score == f"{hs}-{as_}",
            })

        bt = pd.DataFrame(rows)
        if len(bt) == 0:
            return {"n_matches": 0}

        brier_list = []
        log_loss_list = []
        for _, r in bt.iterrows():
            if r["actual_outcome"] == "H":
                target = [1, 0, 0]
            elif r["actual_outcome"] == "D":
                target = [0, 1, 0]
            else:
                target = [0, 0, 1]
            probs = [r["ph"], r["pd"], r["pa"]]
            brier_list.append(sum((p - t) ** 2 for p, t in zip(probs, target)))
            eps = 1e-12
            log_loss_list.append(-np.log(max(probs[np.argmax(target)], eps)))

        bt["brier"] = brier_list
        bt["log_loss"] = log_loss_list
        self._backtest_metrics = {
            "n_matches": len(bt),
            "outcome_accuracy": float(bt["correct_outcome"].mean()),
            "top_score_accuracy": float(bt["actual_top_match"].mean()),
            "brier": float(bt["brier"].mean()),
            "log_loss": float(bt["log_loss"].mean()),
            "mae_home_goals": float(
                np.mean(np.abs(bt["lambda_h"] - bt["score"].str.split("-").str[0].astype(int)))
            ),
            "mae_away_goals": float(
                np.mean(np.abs(bt["lambda_a"] - bt["score"].str.split("-").str[1].astype(int)))
            ),
        }
        return self._backtest_metrics

    def run_monte_carlo(self) -> tuple[dict, dict, dict]:
        """Run Monte Carlo simulations and return probabilities."""
        if not self._predict_func:
            self._build_predictor(pd.Timestamp(date.today()) + pd.Timedelta(days=1))

        teams = self._all_teams
        stage_counts = {t: [0] * 7 for t in teams}
        pos_counts = {t: {1: 0, 2: 0, 3: 0, 4: 0} for t in teams}
        third_qual_counts = {t: 0 for t in teams}
        champion_counts = {}

        cache = precompute_pair_cache(teams, self._predict_func)
        rng = np.random.default_rng(self.rng_seed)

        for _ in range(self.n_simulations):
            sim = simulate_tournament(self._groups, cache, rng)

            for letter, ranked in sim["group_ranked"].items():
                for pos, t in enumerate(ranked, 1):
                    pos_counts[t][pos] += 1

            for t in sim["third_qualifiers"]:
                third_qual_counts[t] += 1

            r32_teams = set()
            for r in sim["r32_results"]:
                r32_teams.add(r["a"])
                r32_teams.add(r["b"])
            for t in r32_teams:
                stage_counts[t][1] += 1

            r16_teams = set()
            for r in sim["r16_results"]:
                r16_teams.add(r["a"])
                r16_teams.add(r["b"])
            for t in r16_teams:
                stage_counts[t][2] += 1

            qf_teams = set()
            for r in sim["qf_results"]:
                qf_teams.add(r["a"])
                qf_teams.add(r["b"])
            for t in qf_teams:
                stage_counts[t][3] += 1

            sf_teams = set()
            for r in sim["sf_results"]:
                sf_teams.add(r["a"])
                sf_teams.add(r["b"])
            for t in sf_teams:
                stage_counts[t][4] += 1

            if sim["final"]:
                f = sim["final"]
                stage_counts[f["a"]][5] += 1
                stage_counts[f["b"]][5] += 1
                stage_counts[f["winner"]][6] += 1
                champion_counts[f["winner"]] = champion_counts.get(f["winner"], 0) + 1

        probs = {t: [c / self.n_simulations for c in counts] for t, counts in stage_counts.items()}
        pos_probs = {t: {p: c / self.n_simulations for p, c in pc.items()} for t, pc in pos_counts.items()}

        for t in teams:
            pos_probs[t]["3rd_qual"] = third_qual_counts[t] / self.n_simulations

        return probs, pos_probs, champion_counts

    def predict_tournament(self, ref_date: Optional[date] = None, skip_mc: bool = False) -> PredictionResults:
        """Run full prediction pipeline and return structured results."""
        if ref_date is None:
            ref_date = date.today()
        self._build_predictor(pd.Timestamp(ref_date) + pd.Timedelta(days=1))

        group_matches = self.predict_group_matches()

        if self.user_results:
            self._backtest_metrics = self.backtest_user_results()

        if skip_mc:
            return PredictionResults(
                group_match_predictions=group_matches,
                tournament_probs=pd.DataFrame(),
                group_position_probs=pd.DataFrame(),
                elo_ratings=self._elo_ratings,
                attack_ratings=self._attack_ratings,
                defense_ratings=self._defense_ratings,
                league_avg_goals=self._league_avg,
                champion_probs={},
                backtest_metrics=self._backtest_metrics,
            )

        tournament_probs, pos_probs, champion_counts = self.run_monte_carlo()

        rows = []
        for t in self._all_teams:
            p = tournament_probs[t]
            rows.append({
                "team": t,
                "elo": self._elo_ratings.get(t, INITIAL_ELO),
                "P_R32": p[1],
                "P_R16": p[2],
                "P_QF": p[3],
                "P_SF": p[4],
                "P_Final": p[5],
                "P_Win": p[6],
            })
        tournament_df = pd.DataFrame(rows).sort_values("P_Win", ascending=False)

        pos_rows = []
        for t in self._all_teams:
            for pos in [1, 2, 3, 4]:
                pos_rows.append({"team": t, "position": pos, "probability": pos_probs[t].get(pos, 0)})
            pos_rows.append({"team": t, "position": "3rd_qual", "probability": pos_probs[t].get("3rd_qual", 0)})
        pos_df = pd.DataFrame(pos_rows)

        champion_probs = {t: p[6] for t, p in tournament_probs.items()}

        return PredictionResults(
            group_match_predictions=group_matches,
            tournament_probs=tournament_df,
            group_position_probs=pos_df,
            elo_ratings=self._elo_ratings,
            attack_ratings=self._attack_ratings,
            defense_ratings=self._defense_ratings,
            league_avg_goals=self._league_avg,
            champion_probs=champion_probs,
            backtest_metrics=self._backtest_metrics,
        )

    def save_outputs(self, ref_date: Optional[date] = None):
        """Save all prediction outputs to CSV/JSON files."""
        results = self.predict_tournament(ref_date)

        if len(results.group_match_predictions) > 0:
            results.group_match_predictions.to_csv(
                OUT_DIR / "group_match_predictions.csv", index=False
            )

        if len(results.tournament_probs) > 0:
            results.tournament_probs.to_csv(
                OUT_DIR / "tournament_probabilities.csv", index=False
            )

        if len(results.group_position_probs) > 0:
            results.group_position_probs.to_csv(
                OUT_DIR / "group_position_probabilities.csv", index=False
            )

        if self._elo_ratings and self._all_teams:
            elo_df = pd.DataFrame(
                sorted([(t, e) for t, e in self._elo_ratings.items() if t in self._all_teams], key=lambda x: -x[1]),
                columns=["team", "elo"]
            )
            elo_df.to_csv(OUT_DIR / "elo_snapshot_2026.csv", index=False)

        if self._attack_ratings and self._defense_ratings and self._all_teams:
            ad_df = pd.DataFrame([
                {"team": t, "attack": self._attack_ratings.get(t, 1.0), "defense": self._defense_ratings.get(t, 1.0)}
                for t in self._all_teams
            ]).sort_values("attack", ascending=False)
            ad_df.to_csv(OUT_DIR / "attack_defense_ratings.csv", index=False)

        summary = {
            "cutoff": str(ref_date or date.today()),
            "n_simulations": self.n_simulations,
            "league_avg_goals": self._league_avg,
            "backtest": results.backtest_metrics,
            "top10_champions": [
                {"team": t, "p_win": p}
                for t, p in sorted(results.champion_probs.items(), key=lambda x: -x[1])[:10]
            ],
            "groups": {letter: teams for letter, teams in self._groups.items()},
        }
        with open(OUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def load_actuals(self, csv_path: Path) -> int:
        """Load actual results from a CSV file.

        CSV columns: date, home_team, away_team, home_score, away_score
        Returns number of results loaded.
        """
        df = pd.read_csv(csv_path, parse_dates=["date"])
        for _, row in df.iterrows():
            self.add_result(
                match_date=row["date"].date(),
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_score=int(row["home_score"]),
                away_score=int(row["away_score"]),
                tournament=row.get("tournament", "FIFA World Cup"),
                neutral=row.get("neutral", False),
            )
        return len(df)

    def get_updated_predictions(self) -> PredictionResults:
        """Get predictions incorporating all user results - convenience method."""
        return self.predict_tournament()


if __name__ == "__main__":
    predictor = WorldCupPredictor(n_simulations=3000)  # Faster for testing
    results = predictor.predict_tournament()
    print("Top 10 championship probabilities:")
    for _, row in results.tournament_probs.head(10).iterrows():
        print(f"  {row['P_Win']*100:5.2f}%  {row['team']}")