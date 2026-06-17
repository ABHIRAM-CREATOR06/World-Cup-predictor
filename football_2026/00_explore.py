"""Quick exploration of the dataset."""
import pandas as pd
import numpy as np

DATA = r"C:\Users\abhir\Downloads\football_analysis"

results = pd.read_csv(f"{DATA}/results.csv", parse_dates=["date"])
goals = pd.read_csv(f"{DATA}/goalscorers.csv", parse_dates=["date"])
shootouts = pd.read_csv(f"{DATA}/shootouts.csv", parse_dates=["date"])
former = pd.read_csv(f"{DATA}/former_names.csv", parse_dates=["start_date", "end_date"])

print("=== results ===")
print(f"rows: {len(results):,}")
print(f"date range: {results['date'].min().date()} -> {results['date'].max().date()}")
print(f"nulls: {results.isna().sum().to_dict()}")
print(f"tournaments: {results['tournament'].nunique()}")
print(f"teams: {pd.concat([results['home_team'], results['away_team']]).nunique()}")

# 2026 WC fixtures
wc26 = results[(results["tournament"] == "FIFA World Cup") & (results["date"].dt.year == 2026)].copy()
print("\n=== 2026 FIFA World Cup fixtures ===")
print(f"rows: {len(wc26)}")
print(f"distinct teams: {pd.concat([wc26['home_team'], wc26['away_team']]).nunique()}")
print(f"date range: {wc26['date'].min().date()} -> {wc26['date'].max().date()}")
print(f"cities: {wc26['city'].unique()}")
print(f"countries: {wc26['country'].unique()}")

# Show all 2026 WC matches
wc26_sorted = wc26.sort_values(["date", "city"])
print("\n=== all 2026 WC matches ===")
for _, r in wc26_sorted.iterrows():
    print(f"{r['date'].date()}  {r['home_team']:20s} vs {r['away_team']:20s}  {r['city']}")

# Find latest match with actual score
played = results.dropna(subset=["home_score", "away_score"])
print(f"\nLatest match with score: {played['date'].max()}")
print(f"Latest FIFA World Cup with score: {played[played['tournament']=='FIFA World Cup']['date'].max()}")
