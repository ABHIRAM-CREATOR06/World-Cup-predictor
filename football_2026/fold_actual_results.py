"""Fold the 16 actual results from Match Days 1-2 of 2026 WC into the dataset,
recompute ELO + attack/defense ratings, and rerun the simulator for the
remaining matches.

Results provided by the user (parsed from text):
  Fri Jun 12:  Mexico 2-0 South Africa    (Group A)
               South Korea 2-1 Czechia   (Group A)
  Sat Jun 13:  Canada 1-1 Bosnia & H.    (Group B)
               USA 4-1 Paraguay           (Group D)
  Sun Jun 14:  Qatar 1-1 Switzerland     (Group B)
               Brazil 1-1 Morocco        (Group C)
               Haiti 0-1 Scotland        (Group C)
               Australia 2-0 Turkiye     (Group D)
               Germany 7-1 Curacao       (Group E)
  Yesterday:   Netherlands 2-2 Japan     (Group F)
               Ivory Coast 1-0 Ecuador   (Group E)
               Sweden 5-1 Tunisia        (Group F)
               Spain 0-0 Cape Verde      (Group H)
  Today:       Belgium 1-1 Egypt         (Group G)
               Saudi Arabia 1-1 Uruguay  (Group H)
               Iran 2-2 New Zealand      (Group G)
"""
import pandas as pd
from pathlib import Path

# Note: dates reconstructed from the user's text. The actual schedule PDF
# says 2026-06-11 was the opener (Mexico vs South Africa), so these are
# likely the 2026-06-12 to 2026-06-15 window.
RESULTS_2026 = [
    # date,        home,                away,         hs, as, tournament,         city,        country,         neutral
    ("2026-06-11", "Mexico",            "South Africa", 2, 0, "FIFA World Cup", "Mexico City", "Mexico", False),
    ("2026-06-11", "South Korea",       "Czech Republic", 2, 1, "FIFA World Cup", "Guadalupe", "Mexico", False),
    ("2026-06-12", "Canada",            "Bosnia and Herzegovina", 1, 1, "FIFA World Cup", "Toronto", "Canada", False),
    ("2026-06-12", "United States",     "Paraguay", 4, 1, "FIFA World Cup", "Inglewood", "United States", False),
    ("2026-06-13", "Qatar",             "Switzerland", 1, 1, "FIFA World Cup", "Santa Clara", "United States", False),
    ("2026-06-13", "Brazil",            "Morocco", 1, 1, "FIFA World Cup", "East Rutherford", "United States", False),
    ("2026-06-13", "Haiti",             "Scotland", 0, 1, "FIFA World Cup", "Foxborough", "United States", False),
    ("2026-06-13", "Australia",         "Turkey", 2, 0, "FIFA World Cup", "Vancouver", "Canada", False),
    ("2026-06-13", "Germany",           "Curaçao", 7, 1, "FIFA World Cup", "Houston", "United States", False),
    ("2026-06-14", "Netherlands",       "Japan", 2, 2, "FIFA World Cup", "Guadalupe", "Mexico", False),
    ("2026-06-14", "Ivory Coast",       "Ecuador", 1, 0, "FIFA World Cup", "Philadelphia", "United States", False),
    ("2026-06-14", "Sweden",            "Tunisia", 5, 1, "FIFA World Cup", "Guadalupe", "Mexico", False),
    ("2026-06-14", "Spain",             "Cape Verde", 0, 0, "FIFA World Cup", "Atlanta", "United States", False),
    ("2026-06-15", "Belgium",           "Egypt", 1, 1, "FIFA World Cup", "Inglewood", "United States", False),
    ("2026-06-15", "Saudi Arabia",      "Uruguay", 1, 1, "FIFA World Cup", "Miami Gardens", "United States", False),
    ("2026-06-15", "Iran",              "New Zealand", 2, 2, "FIFA World Cup", "Inglewood", "United States", False),
]

df = pd.DataFrame(RESULTS_2026, columns=["date", "home_team", "away_team", "home_score", "away_score", "tournament", "city", "country", "neutral"])
df["date"] = pd.to_datetime(df["date"])
df["neutral"] = df["neutral"].astype(bool)
print(f"{len(df)} actual results to fold in")
print(df.head())

OUT = Path(r"C:\Users\abhir\.mavis\sessions\mvs_fbdb1a4d0ee34fc9b22f88407249a52a\workspace\football_2026\out")
df.to_csv(OUT / "actual_results_2026.csv", index=False)
print(f"Saved to {OUT / 'actual_results_2026.csv'}")
