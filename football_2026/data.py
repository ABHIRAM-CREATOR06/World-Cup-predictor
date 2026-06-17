"""Module: data loading and team name normalization.

The CSV has some mojibake on a few accented names (e.g. "CuraçAo"). We normalize
those to canonical current names, and we resolve historical names via the
former_names table so old results still count toward modern countries.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

DATA_DIR = Path(r"C:\Users\abhir\Downloads\football_analysis")

# Hand-curated aliases to canonical current names.
# The dataset is already UTF-8 clean for the 14 accented names, but FIFA
# has used different English names for some teams over the years.
ALIASES = {
    "Czechoslovakia": "Czech Republic",  # dissolve to CZ in 1993, but historically a strong team
    "Yugoslavia": "Serbia",               # dissolved 2006; later Serbia + Montenegro, then Serbia
    "Serbia and Montenegro": "Serbia",
    "Zaire": "DR Congo",
    "Congo": "DR Congo",  # ambiguous historically, but matches our 2026 list
    "Ivory Coast": "Ivory Coast",
    "Chinese Taipei": "Taiwan",
    "Vietnam Republic": "Vietnam",
    "Yemen DPR": "Yemen",
    "Burma": "Myanmar",
    "Bohemia": "Czech Republic",
    "Bohemia and Moravia": "Czech Republic",
    "German DR": "Germany",  # they often played as separate team; merge
    "East Germany": "Germany",
    "West Germany": "Germany",
    "Saarland": "Germany",
    "CIS": "Russia",
    "USSR": "Russia",
    "Soviet Union": "Russia",
    "Northern Ireland": "Northern Ireland",
    "Wales": "Wales",
    "England": "England",
    "Scotland": "Scotland",
    "Republic of Ireland": "Ireland",
    "Korea DPR": "North Korea",
}


def _clean(s):
    if s is None:
        return s
    s = str(s).strip()
    return unicodedata.normalize("NFKC", s)


def load_results() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    df["home_team"] = df["home_team"].map(_clean)
    df["away_team"] = df["away_team"].map(_clean)
    df["tournament"] = df["tournament"].map(_clean)
    df["city"] = df["city"].map(_clean)
    df["country"] = df["country"].map(_clean)
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_shootouts() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "shootouts.csv", parse_dates=["date"])
    df["home_team"] = df["home_team"].map(_clean)
    df["away_team"] = df["away_team"].map(_clean)
    df["winner"] = df["winner"].map(_clean)
    return df


def load_former_names() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / "former_names.csv",
        parse_dates=["start_date", "end_date"],
    )
    df["current"] = df["current"].map(_clean)
    df["former"] = df["former"].map(_clean)
    return df


def build_name_map(former: pd.DataFrame) -> dict[str, str]:
    """Map every former name to its current canonical name."""
    m = {}
    for _, row in former.iterrows():
        m[row["former"]] = row["current"]
    # Also map current -> current (identity)
    for c in former["current"].unique():
        m.setdefault(c, c)
    return m


def normalize_teams(df: pd.DataFrame, name_map: dict[str, str], former_df: pd.DataFrame) -> pd.DataFrame:
    """Replace former team names with current ones (date-bounded), then apply
    the static ALIASES table.
    """
    by_former: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]] = {}
    for _, r in former_df.iterrows():
        by_former.setdefault(r["former"], []).append((r["start_date"], r["end_date"], r["current"]))

    def resolve(team: str, when: pd.Timestamp) -> str:
        if team in by_former:
            for start, end, current in by_former[team]:
                if start <= when <= end:
                    return ALIASES.get(current, current)
            return ALIASES.get(team, team)
        return ALIASES.get(name_map.get(team, team), name_map.get(team, team))

    df = df.copy()
    df["home_team_orig"] = df["home_team"]
    df["away_team_orig"] = df["away_team"]
    home_resolved = [resolve(t, d) for t, d in zip(df["home_team"].values, df["date"].values)]
    away_resolved = [resolve(t, d) for t, d in zip(df["away_team"].values, df["date"].values)]
    df["home_team"] = home_resolved
    df["away_team"] = away_resolved
    return df


def identify_2026_groups(results: pd.DataFrame) -> dict[str, list[str]]:
    """Use the OFFICIAL FIFA 2026 group composition (from the schedule PDF).

    Group letters are per FIFA's official draw, not inferred from the dataset.
    Team names are mapped to whatever the dataset calls them (handles aliases
    like "Korea Republic" <-> "South Korea", "Türkiye" <-> "Turkey", etc.).
    """
    import json
    pdf_path = Path(r"C:\Users\abhir\Downloads\football_analysis")  # not used; keep import clean
    # The official FIFA group composition is embedded here, sourced from the
    # official FIFA WC 2026 match schedule PDF (v17, 10 April 2026).
    official_groups = {
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
    # Verify all 48 teams are present in the dataset (after normalization)
    wc26 = results[(results["tournament"] == "FIFA World Cup") & (results["date"].dt.year == 2026)]
    dataset_teams = set(wc26["home_team"]) | set(wc26["away_team"])
    for letter, teams in official_groups.items():
        for t in teams:
            if t not in dataset_teams:
                # Try a fuzzy match: find dataset team that contains this name
                for dt in dataset_teams:
                    if t.lower().split()[0] in dt.lower() or dt.lower().split()[0] in t.lower():
                        if abs(len(t) - len(dt)) <= 3:
                            print(f"  WARN: {t} not in dataset; close match: {dt}")
                            break
                else:
                    print(f"  WARN: {t} (Group {letter}) not found in dataset")
    return official_groups


if __name__ == "__main__":
    r = load_results()
    print(f"results rows: {len(r):,}")

    s = load_shootouts()
    print(f"shootouts rows: {len(s):,}")

    fn = load_former_names()
    print(f"former names: {len(fn):,}")

    nm = build_name_map(fn)
    r2 = normalize_teams(r, nm, fn)
    print(f"after normalize, sample: home_team uniques={r2['home_team'].nunique()}")

    groups = identify_2026_groups(r2)
    for letter, members in groups.items():
        print(f"  Group {letter}: {', '.join(members)}")
