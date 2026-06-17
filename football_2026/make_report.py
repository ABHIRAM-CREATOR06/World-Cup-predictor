"""Generate a comprehensive markdown report combining:
  - ELO + ratings post-MD2
  - Backtest of the 16 actual results
  - Updated group standings
  - Remaining match predictions
  - Tournament progression probabilities
  - Most-likely bracket
"""
import pandas as pd
from pathlib import Path

OUT = Path(r"C:\Users\abhir\.mavis\sessions\mvs_fbdb1a4d0ee34fc9b22f88407249a52a\workspace\football_2026\out")

elo = pd.read_csv(OUT / "elo_snapshot_2026_post_md2.csv")
bt = pd.read_csv(OUT / "backtest_md1_2.csv")
probs = pd.read_csv(OUT / "tournament_probabilities.csv")
positions = pd.read_csv(OUT / "group_position_probabilities.csv")
group_preds = pd.read_csv(OUT / "group_match_predictions.csv")

# Build report
md = []
md.append("# 2026 FIFA World Cup — Predictive Model Report (post-Match Day 2)\n")
md.append("**Model**: ELO (FiveThirtyEight-style) + attack/defense ratings + Poisson goal model.\n")
md.append("**Bracket**: Official FIFA 2026 R32 bracket (from schedule PDF v17).\n")
md.append("**Data**: international results 1872 → 2026-06-15 (49,437 matches including 16 actual MD1-2 results).\n")
md.append("**Backtested** on 2018 WC, Euro 2020, 2022 WC, Euro 2024, Copa America 2024: Brier 0.55-0.65, accuracy 45-56%.\n")
md.append("**Monte Carlo**: 20,000 full tournament simulations from the post-MD2 state.\n")
md.append("")

# Section 1: ELO snapshot
md.append("## 1. Top 15 ELO (post-MD2, 2026-06-16)\n")
md.append("| Rank | Team | ELO |")
md.append("|---:|---|---:|")
for i, (_, r) in enumerate(elo.head(15).iterrows(), 1):
    md.append(f"| {i} | {r['team']} | {r['elo']:.0f} |")
md.append("")

# Section 2: Backtest on actual matches
md.append("## 2. Backtest on the 16 actual results (MD1-2)\n")
md.append("How the model would have predicted each match, using only data from before the match.\n")
md.append("| Date | Match | Score | P(H) | P(D) | P(A) | Pred | Actual | ✓ |")
md.append("|---|---|---|---:|---:|---:|---|---|---|")
for _, r in bt.iterrows():
    md.append(f"| {r['date']} | {r['match']} | {r['score']} | {r['ph']:.2f} | {r['pd']:.2f} | {r['pa']:.2f} | {r['pred_outcome']} | {r['actual_outcome']} | {'✓' if r['correct_outcome'] else '✗'} |")
md.append("")

n = len(bt)
md.append(f"**Outcome accuracy: {bt['correct_outcome'].mean()*100:.1f}%** ({int(bt['correct_outcome'].sum())}/{n} correct)\n")
md.append(f"**Mean Brier score: {bt['brier'].mean():.3f}** (random baseline: 0.667, perfect: 0.0)\n")
md.append(f"**Mean log loss: {bt['log_loss'].mean():.3f}** (random baseline: 1.099, perfect: 0.0)\n")
md.append("")

# Section 3: Current group standings
md.append("## 3. Current group standings (after MD1-2)\n")
md.append("| Group | 1st | 2nd | 3rd | 4th |")
md.append("|---|---|---|---|---|")
groups_fifa = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
played = pd.read_csv(OUT / "actual_results_2026.csv", parse_dates=["date"])
for letter in "ABCDEFGHIJKL":
    teams = groups_fifa[letter]
    pts = {t: 0 for t in teams}; gf = {t: 0 for t in teams}; ga = {t: 0 for t in teams}
    gp = played[played["home_team"].isin(teams) | played["away_team"].isin(teams)]
    for _, m in gp.iterrows():
        h, a = m["home_team"], m["away_team"]
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        if hs > as_: pts[h] += 3
        elif hs < as_: pts[a] += 3
        else: pts[h] += 1; pts[a] += 1
        gf[h] += hs; gf[a] += as_; ga[h] += as_; ga[a] += hs
    ranked = sorted(teams, key=lambda t: (-pts[t], -(gf[t]-ga[t]), -gf[t]))
    md.append(f"| {letter} | {ranked[0]} {pts[ranked[0]]}p | {ranked[1]} {pts[ranked[1]]}p | {ranked[2]} {pts[ranked[2]]}p | {ranked[3]} {pts[ranked[3]]}p |")
md.append("")

# Section 4: Remaining predictions
md.append("## 4. Predictions for remaining group matches\n")
md.append("P(H)/P(D)/P(A) are win/draw/loss probabilities. λ values are expected goals.\n")
md.append("| Group | Match | P(H) | P(D) | P(A) | λ_H | λ_A | Top score |")
md.append("|---|---|---:|---:|---:|---:|---:|---|")
for _, r in group_preds.sort_values(["group", "home"]).iterrows():
    md.append(f"| {r['group']} | {r['home']} vs {r['away']} | "
              f"{r['ph']:.2f} | {r['pd']:.2f} | {r['pa']:.2f} | "
              f"{r['lambda_home']:.2f} | {r['lambda_away']:.2f} | "
              f"{r['most_likely_score']} ({r['most_likely_prob']:.2f}) |")
md.append("")

# Section 5: Tournament progression
md.append("## 5. Tournament progression probabilities\n")
md.append("| Team | ELO | P(R32) | P(R16) | P(QF) | P(SF) | P(Final) | P(Win) |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for _, r in probs.iterrows():
    md.append(f"| {r['team']} | {r['elo']:.0f} | {r['P_R32']:.2f} | {r['P_R16']:.2f} | {r['P_QF']:.2f} | {r['P_SF']:.2f} | {r['P_Final']:.2f} | {r['P_Win']:.2f} |")
md.append("")

# Section 6: Top 10
md.append("## 6. Top 10 most likely champions\n")
md.append("| Rank | Team | P(Win) |")
md.append("|---:|---|---:|")
for i, (_, r) in enumerate(probs.head(10).iterrows(), 1):
    md.append(f"| {i} | {r['team']} | {r['P_Win']*100:.1f}% |")
md.append("")

# Notes
md.append("## 7. Notes\n")
md.append(
    "- **The model is now calibrated against actual MD1-2 results.** ELO has been updated for all 16 matches; attack/defense ratings incorporate the actual goals.\n"
    "- **The big movers**: Brazil's ELO dropped slightly after drawing Morocco; Spain's dropped a bit after the Cape Verde draw; Germany's shot up after the 7-1; USA's ELO rose after the 4-1 win.\n"
    "- **Morocco still rated top** - they didn't lose (drew Brazil 1-1) and the model weights their 2022 WC run heavily.\n"
    "- **Group H is wide open**: Spain, Uruguay, and Saudi Arabia all on 1pt. Any of them could win the group.\n"
    "- **Backtest is honest** - these are the model's pre-match predictions, not the model fitted to outcomes. 37.5% outcome accuracy on this batch is below the long-run average (50-55%) because MD1-2 was unusually draw-heavy (10 of 16 ended in draws).\n"
    "- **Draws hurt outcome accuracy directly**: a 0-0 outcome and 3-2 outcome both count as 'wrong' if the model predicted 1-0. The 1.14 log-loss reflects this.\n"
)

with open(OUT / "report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"Saved {OUT / 'report.md'}")
