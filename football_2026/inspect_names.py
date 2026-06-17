"""Inspect what team names actually appear in the dataset."""
import pandas as pd
import unicodedata

df = pd.read_csv(r"C:\Users\abhir\Downloads\football_analysis\results.csv", encoding="utf-8")
all_teams = sorted(set(df["home_team"].tolist() + df["away_team"].tolist()))

# Look for names with non-ASCII or accent issues
suspect = [t for t in all_teams if any(ord(c) > 127 for c in t)]
print(f"teams with non-ASCII ({len(suspect)}):")
for t in suspect:
    cps = " ".join(f"{ord(c):04x}" for c in t)
    print(f"  {t!r}  ({cps})")

# Also: any name with ?, broken char, or weird punct
weird = [t for t in all_teams if any(c in t for c in "?\u2018\u2019\u201A\u201C\u201D\u00a0")]
print(f"\nteams with smart-quote/special ({len(weird)}):")
for t in weird[:30]:
    print(f"  {t!r}")

# Top aliases - some countries have multiple names
aliases = ["Korea", "Iran", "Cura", "Côte", "Cote", "DR", "Verde", "Tom", "Prín", "Princ", "Swazi", "Eswa",
           "Cape", "Cabo", "Czechoslov", "Yugo", "Serb", "Monten", "Macedon", "Czech", "Slovak", "Holland",
           "Nether", "UAE", "USA", "Burkina", "Volta", "Benin", "Dahom"]
for a in aliases:
    matches = [t for t in all_teams if a.lower() in t.lower()]
    if matches:
        print(f"\n  '{a}': {matches}")
