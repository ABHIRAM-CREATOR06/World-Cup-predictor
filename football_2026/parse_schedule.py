"""Parse the schedule PDF: extract the official FIFA group assignments, knockout
bracket (R32, R16, QF, SF, F), with match numbers, dates, kickoff times,
venues, and team codes.

This supersedes my "union-find" group detection from the dataset.
"""
import pdfplumber
import re
import json
import pandas as pd
from pathlib import Path

PDF = Path(r"C:\Users\abhir\Downloads\FWC26 Match Schedule_v17_10042026_EN.pdf")
OUT = Path(r"C:\Users\abhir\.mavis\sessions\mvs_fbdb1a4d0ee34fc9b22f88407249a52a\workspace\football_2026")

with pdfplumber.open(PDF) as pdf:
    page = pdf.pages[0]
    tables = page.extract_tables()

# Table 2: group compositions
group_table = tables[2]
# Each row has 6 cells: the 12 groups are split across 2 rows
groups_fifa = {}
for row in group_table:
    if not row:
        continue
    # The cells contain "GROUP X\nTEAM1\nTEAM2\nTEAM3\nTEAM4"
    for cell in row:
        if cell and "GROUP" in cell:
            lines = [l.strip() for l in cell.split("\n") if l.strip()]
            if len(lines) >= 5:
                letter = lines[0].replace("GROUP", "").strip()[0]
                teams = []
                for line in lines[1:]:
                    # "MEXICO (MEX)" -> "Mexico" via the team name before the parenthesis
                    name = re.sub(r"\s*\([A-Z]{3}\)\s*", "", line).strip()
                    if name:
                        teams.append(name)
                if len(teams) == 4:
                    groups_fifa[letter] = teams

print("=== FIFA Official Group Composition ===")
for letter in "ABCDEFGHIJKL":
    if letter in groups_fifa:
        print(f"  Group {letter}: {groups_fifa[letter]}")

# Table 1: matches (very wide). Each row is a venue; cells are match blocks.
# A match block looks like: "6 00:00\nAUS\nv\nTUR\nD"
# where 6 = match number, 00:00 = kickoff, AUS/TUR = 3-letter codes, D = group letter

# Day headers are in a separate row; let me extract days from the first table
# But Table 1 also has day headers at the bottom (in reverse order: 11 June Thursday, etc.)
# Day names were in the original extraction: "Thursday 11", "Friday 12", etc.

# Let me also extract venue rows: VANCOUVER, SEATTLE, SAN FRANCISCO BAY AREA, ...
venue_order = ["VANCOUVER", "SEATTLE", "SAN FRANCISCO BAY AREA", "LOS ANGELES",
               "GUADALAJARA", "MEXICO CITY", "MONTERREY", "HOUSTON", "DALLAS",
               "KANSAS CITY", "ATLANTA", "MIAMI", "TORONTO", "BOSTON", "NEW YORK NEW JERSEY"]

# Save what we have
out_data = {
    "fifa_groups": groups_fifa,
}
with open(OUT / "fifa_groups.json", "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2)
print(f"\nSaved {OUT / 'fifa_groups.json'}")
