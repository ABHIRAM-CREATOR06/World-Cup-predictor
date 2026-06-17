"""Render the most-likely bracket as an HTML page + PNG, with the OFFICIAL FIFA 2026 bracket structure."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = Path(r"C:\Users\abhir\.mavis\sessions\mvs_fbdb1a4d0ee34fc9b22f88407249a52a\workspace\football_2026\out")
sys.path.insert(0, str(OUT.parent))

with open(OUT / "most_likely_bracket.json") as f:
    b = json.load(f)

r32 = b["R32"]
r16 = b["R16"]
qf = b["QF"]
sf = b["SF"]
final = b["final"]
third_place = b.get("third_place")

# === HTML report ===
group_html = '<div style="background:#161b22;padding:20px;border-radius:8px;margin-top:30px;">\n'
group_html += '<h2 style="color:#e6edf3;margin-top:0;">Group Stage (12 groups, 48 teams — top 2 + 8 best 3rd advance)</h2>\n'
group_html += '<p style="color:#8b949e;font-size:13px;">Group letters and team composition per the official FIFA draw.</p>\n'
group_html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">\n'
for letter in "ABCDEFGHIJKL":
    rk = b["most_likely_group_standings"][letter]
    qual_third = any(letter == x[0] for x in b["best_8_thirds"])
    group_html += f'<div style="background:#0d1117;padding:10px;border-radius:6px;border:1px solid #30363d;">\n'
    group_html += f'<div style="color:#58a6ff;font-weight:bold;margin-bottom:5px;">Group {letter}</div>\n'
    for pos, t in enumerate(rk, 1):
        if pos <= 2:
            bg, opacity, font_w = "#1f6feb", "1", "600"
        elif pos == 3 and qual_third:
            bg, opacity, font_w = "#2ea043", "1", "600"
        else:
            bg, opacity, font_w = "#6e7681", "0.5", "400"
        label = ["1st", "2nd", "3rd", "4th"][pos-1]
        group_html += f'<div style="background:{bg};padding:5px 8px;margin:2px 0;border-radius:4px;opacity:{opacity};color:#fff;font-size:12px;font-weight:{font_w};">{label}: {t}</div>\n'
    group_html += '</div>\n'
group_html += '</div></div>\n'

# Bracket table HTML
def fmt_match(m, slot_a_key="slot_a", slot_b_key="slot_b"):
    a = m.get("team_a", m.get("a"))
    b = m.get("team_b", m.get("b"))
    sa = m.get("slot_a", "")
    sb = m.get("slot_b", "")
    w = m["winner"]
    a_label = f"{a} ({sa})" if sa else a
    b_label = f"{b} ({sb})" if sb else b
    return a_label, b_label, w

r32_html = '<div style="background:#161b22;padding:20px;border-radius:8px;margin-top:30px;">\n'
r32_html += '<h2 style="color:#e6edf3;margin-top:0;">Round of 32 (28 June – 3 July 2026) — Official FIFA Bracket</h2>\n'
r32_html += '<table style="width:100%;border-collapse:collapse;color:#e6edf3;font-size:13px;">\n'
r32_html += '<tr style="border-bottom:1px solid #30363d;"><th style="text-align:left;padding:6px;">Match</th><th style="text-align:left;padding:6px;">Date</th><th style="text-align:left;padding:6px;">Home</th><th style="text-align:left;padding:6px;">Away</th><th style="text-align:left;padding:6px;">Winner</th></tr>\n'
for m in r32:
    a, b, w = fmt_match(m)
    r32_html += f'<tr style="border-bottom:1px solid #21262d;"><td style="padding:5px 6px;">M{m["match_num"]}</td><td style="padding:5px 6px;">{m.get("date","")}</td><td style="padding:5px 6px;">{a}</td><td style="padding:5px 6px;">{b}</td><td style="padding:5px 6px;color:#3fb950;font-weight:600;">{w}</td></tr>\n'
r32_html += '</table></div>\n'

# Knockout bracket
def fmt_simple(m, ka="team_a", kb="team_b"):
    return m[ka], m[kb], m["winner"]

ko_html = '<div style="background:#161b22;padding:20px;border-radius:8px;margin-top:30px;">\n'
ko_html += '<h2 style="color:#e6edf3;margin-top:0;">Knockout Bracket (Most Likely Path)</h2>\n'
ko_html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:15px;">\n'
ko_html += '<div><h3 style="color:#58a6ff;">Round of 16 (4-7 Jul)</h3><ul style="color:#e6edf3;">'
for m in r16:
    a, b, w = fmt_simple(m)
    ko_html += f'<li>R16-{m["r16_idx"]}: {a} vs {b} → <b style="color:#3fb950;">{w}</b></li>'
ko_html += '</ul></div>\n'
ko_html += '<div><h3 style="color:#58a6ff;">Quarterfinals (9-11 Jul)</h3><ul style="color:#e6edf3;">'
for m in qf:
    a, b, w = fmt_simple(m)
    ko_html += f'<li>QF-{m["qf_idx"]}: {a} vs {b} → <b style="color:#3fb950;">{w}</b></li>'
ko_html += '</ul></div>\n'
ko_html += '<div><h3 style="color:#58a6ff;">Semifinals (14-15 Jul)</h3><ul style="color:#e6edf3;">'
for m in sf:
    a, b, w = fmt_simple(m)
    ko_html += f'<li>SF-{m["sf_idx"]}: {a} vs {b} → <b style="color:#3fb950;">{w}</b></li>'
ko_html += '</ul></div>\n'
if third_place:
    ko_html += f'<div><h3 style="color:#58a6ff;">3rd Place (18 Jul)</h3><ul style="color:#e6edf3;"><li>{third_place["team_a"]} vs {third_place["team_b"]} → <b style="color:#3fb950;">{third_place["winner"]}</b></li></ul></div>\n'
ko_html += '</div>'
ko_html += f'<div style="text-align:center;margin-top:20px;padding:20px;background:#0d1117;border:2px solid #3fb950;border-radius:8px;"><div style="color:#3fb950;font-size:24px;font-weight:bold;">🏆 {final["winner"]}</div><div style="color:#8b949e;font-size:14px;margin-top:5px;">Final, 19 July 2026, Philadelphia — {final["team_a"]} vs {final["team_b"]}</div></div>\n'
ko_html += '</div>\n'

# Top 10 championship
import pandas as pd
probs_df = pd.read_csv(OUT / "tournament_probabilities.csv")
top10 = probs_df.head(10)
top10_html = '<div style="background:#161b22;padding:20px;border-radius:8px;margin-top:30px;">\n'
top10_html += '<h2 style="color:#e6edf3;margin-top:0;">Top 10 Championship Probabilities (Monte Carlo, 20,000 sims)</h2>\n'
top10_html += '<table style="width:100%;border-collapse:collapse;color:#e6edf3;">\n'
top10_html += '<tr style="border-bottom:1px solid #30363d;"><th style="text-align:left;padding:8px;">Rank</th><th style="text-align:left;padding:8px;">Team</th><th style="text-align:right;padding:8px;">P(Win)</th><th style="text-align:right;padding:8px;">P(Final)</th><th style="text-align:right;padding:8px;">P(SF)</th><th style="text-align:right;padding:8px;">P(R32)</th></tr>\n'
for i, (_, row) in enumerate(top10.iterrows(), 1):
    top10_html += f'<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px 8px;">{i}</td><td style="padding:6px 8px;font-weight:600;">{row["team"]}</td><td style="padding:6px 8px;text-align:right;color:#3fb950;">{row["P_Win"]*100:.1f}%</td><td style="padding:6px 8px;text-align:right;">{row["P_Final"]*100:.1f}%</td><td style="padding:6px 8px;text-align:right;">{row["P_SF"]*100:.1f}%</td><td style="padding:6px 8px;text-align:right;">{row["P_R32"]*100:.1f}%</td></tr>\n'
top10_html += '</table></div>\n'

# Backtest
bt_df = pd.read_csv(OUT / "backtest_summary.csv")
bt_html = '<div style="background:#161b22;padding:20px;border-radius:8px;margin-top:30px;">\n'
bt_html += '<h2 style="color:#e6edf3;margin-top:0;">Model Validation — Backtest on Past Tournaments</h2>\n'
bt_html += '<p style="color:#8b949e;font-size:13px;">Predicting each tournament from data strictly before its start. Lower Brier / RPS = better. Outcome accuracy = % correct W/D/L picks.</p>\n'
bt_html += '<table style="width:100%;border-collapse:collapse;color:#e6edf3;">\n'
bt_html += '<tr style="border-bottom:1px solid #30363d;"><th style="text-align:left;padding:8px;">Tournament</th><th style="text-align:right;padding:8px;">Matches</th><th style="text-align:right;padding:8px;">Brier ↓</th><th style="text-align:right;padding:8px;">Log Loss ↓</th><th style="text-align:right;padding:8px;">RPS ↓</th><th style="text-align:right;padding:8px;">Accuracy ↑</th></tr>\n'
for _, row in bt_df.iterrows():
    bt_html += f'<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px 8px;">{row["tournament"]}</td><td style="padding:6px 8px;text-align:right;">{int(row["n_matches"])}</td><td style="padding:6px 8px;text-align:right;">{row["brier"]:.3f}</td><td style="padding:6px 8px;text-align:right;">{row["log_loss"]:.3f}</td><td style="padding:6px 8px;text-align:right;">{row["rps"]:.3f}</td><td style="padding:6px 8px;text-align:right;color:#3fb950;">{row["outcome_accuracy"]*100:.1f}%</td></tr>\n'
bt_html += '</table></div>\n'

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>2026 World Cup Prediction</title>
<style>
body {{ background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 30px; }}
.container {{ max-width: 1700px; margin: 0 auto; }}
h1 {{ text-align: center; color: #e6edf3; margin-bottom: 5px; }}
.subtitle {{ text-align: center; color: #8b949e; margin-bottom: 30px; }}
ul li {{ margin: 4px 0; }}
</style>
</head>
<body>
<div class="container">
<h1>2026 FIFA World Cup — Predictive Model</h1>
<p class="subtitle">ELO + Poisson · Monte Carlo 20,000 sims · Official FIFA bracket (PDF v17)</p>

{group_html}
{r32_html}
{ko_html}
{top10_html}
{bt_html}

<div style="background:#161b22;padding:20px;border-radius:8px;margin-top:30px;color:#8b949e;font-size:13px;">
<h2 style="color:#e6edf3;margin-top:0;">Methodology & Limitations</h2>
<p><b>Model:</b> FiveThirtyEight-style ELO (goal-difference modifier) + recency-weighted attack/defense ratings (2-year half-life) + Poisson goal model with home/neutral adjustment.</p>
<p><b>Bracket:</b> The R32, R16, QF, SF, and 3rd-place match all follow the OFFICIAL FIFA 2026 structure from the schedule PDF. The 8 best 3rd-placed teams fill the 3rd-slots via bipartite matching that respects FIFA's per-slot eligibility constraints (e.g., slot M76 can only take 3rds from C/E/F/H/I).</p>
<p><b>Knockouts:</b> Regulation draw rate is halved to reflect extra time + penalties resolving many regulation ties. Penalty outcomes are weighted by regulation win expectancy.</p>
<p><b>Limitations:</b> No individual player data — star players, injuries, manager changes, and locker-room dynamics are not captured. The recency weighting (2-year half-life) over-credits recent form: Morocco rated higher than ELO would suggest due to 2022 WC run.</p>
</div>

</div>
</body>
</html>"""

with open(OUT / "predictions.html", "w", encoding="utf-8") as f:
    f.write(html)
print(f"Saved {OUT / 'predictions.html'}")

# === PNG bracket (matplotlib) ===
fig, ax = plt.subplots(figsize=(20, 16))
ax.set_xlim(0, 8)
ax.set_ylim(0, 20)
ax.axis("off")
ax.set_facecolor("#0d1117")
fig.patch.set_facecolor("#0d1117")

# 16 R32 on left, then 8 R16, 4 QF, 2 SF, 1 Final
def y_for(round_idx, match_idx):
    counts = [16, 8, 4, 2, 1]
    total = counts[round_idx]
    return 19 - (match_idx + 0.5) * (18.0 / total)

# Combine all matches in order: R32, R16, QF, SF, Final
all_matches = [r32, r16, qf, sf, [final]]
round_labels = ["Round of 32 (28 Jun - 3 Jul)", "Round of 16 (4-7 Jul)", "Quarterfinals (9-11 Jul)", "Semifinals (14-15 Jul)", "Final (19 Jul)"]

for ri, matches in enumerate(all_matches):
    x_left = ri * 1.5
    for mi, m in enumerate(matches):
        y = y_for(ri, mi)
        team_a = m["team_a"]; team_b = m["team_b"]
        w = m["winner"]
        if ri == 0:
            sa = m.get("slot_a", ""); sb = m.get("slot_b", "")
        else:
            sa = ""; sb = ""
        a_is_winner = w == team_a
        b_is_winner = w == team_b
        # Team A box
        face = "#1f6feb" if a_is_winner else "#6e7681"
        rect_a = mpatches.FancyBboxPatch((x_left, y - 0.18), 1.3, 0.36, boxstyle="round,pad=0.02",
                                          facecolor=face, edgecolor="none", alpha=0.9)
        ax.add_patch(rect_a)
        ax.text(x_left + 0.04, y + 0.06, f"{team_a} ({sa})" if sa else team_a, color="white", fontsize=8,
                fontweight="bold" if a_is_winner else "normal", va="center")
        # Team B box
        face_b = "#1f6feb" if b_is_winner else "#6e7681"
        rect_b = mpatches.FancyBboxPatch((x_left, y - 0.54), 1.3, 0.36, boxstyle="round,pad=0.02",
                                          facecolor=face_b, edgecolor="none", alpha=0.9)
        ax.add_patch(rect_b)
        ax.text(x_left + 0.04, y - 0.31, f"{team_b} ({sb})" if sb else team_b, color="white", fontsize=8,
                fontweight="bold" if b_is_winner else "normal", va="center")
        # Connector
        if ri < len(all_matches) - 1:
            x1 = x_left + 1.3
            next_y = y_for(ri + 1, mi // 2)
            ax.plot([x1, x1 + 0.05, x1 + 0.05, x1 + 0.2],
                    [y, y, next_y, next_y], color="#8b949e", lw=0.6)
    # Round label
    ax.text(ri * 1.5 + 0.65, 19.6, round_labels[ri], color="#58a6ff",
            fontsize=12, fontweight="bold", ha="center")

# Title and champion banner
ax.text(3.5, 19.95, "2026 FIFA World Cup — Most Likely Bracket (Official FIFA Structure)",
        color="white", fontsize=15, fontweight="bold", ha="center")
champion = final["winner"]
ax.text(3.5, 0.4, f"🏆 PREDICTED CHAMPION: {champion}",
        color="#3fb950", fontsize=18, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d1117", edgecolor="#3fb950", lw=2))

plt.tight_layout()
plt.savefig(OUT / "bracket.png", dpi=110, facecolor="#0d1117", bbox_inches="tight")
print(f"Saved {OUT / 'bracket.png'}")
