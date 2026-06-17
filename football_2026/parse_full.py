"""Full schedule parser: groups, knockout dates, and the R32 bracket structure
(fixed bracket positions: who plays who)."""
import pdfplumber
import re
import json
from pathlib import Path

PDF = Path(r"C:\Users\abhir\Downloads\FWC26 Match Schedule_v17_10042026_EN.pdf")
OUT = Path(r"C:\Users\abhir\.mavis\sessions\mvs_fbdb1a4d0ee34fc9b22f88407249a52a\workspace\football_2026")

with pdfplumber.open(PDF) as pdf:
    page = pdf.pages[0]
    tables = page.extract_tables()

# Groups
group_table = tables[2]
groups_fifa = {}
for row in group_table:
    if not row:
        continue
    for cell in row:
        if cell and cell.startswith("GROUP"):
            lines = [l.strip() for l in cell.split("\n") if l.strip()]
            if len(lines) >= 5:
                letter = lines[0].replace("GROUP", "").strip()[0]
                teams = []
                for line in lines[1:]:
                    name = re.sub(r"\s*\([A-Z]{3}\)\s*", "", line).strip()
                    if name:
                        teams.append(name)
                if len(teams) == 4:
                    groups_fifa[letter] = teams

# Group B and H are split across cells in row 1 and 2
# Cell 1 + 2 = Group B (Canada, Bosnia, Qatar, Switzerland)
# Cell 8 + 9 = Group H (Spain, Cabo Verde, Saudi Arabia, Uruguay)
# Let me reconstruct them by combining cells 1+2 and 8+9
all_cells = []
for row in group_table:
    if not row:
        continue
    for cell in row:
        if cell:
            all_cells.append(cell)
# Cell 1: "GROU\nCANA\nBOSNIA\nQATAR\nSWITZ"
# Cell 2: "P B\nDA (CAN)\n& HERZEGOVINA (BIH)\n(QAT)\nERLAND (SUI)"
# Combined line by line: GROU+P B = GROUP B, CANA+DA = CANADA, etc.
def merge_cells(c1, c2):
    l1 = [l.strip() for l in c1.split("\n")]
    l2 = [l.strip() for l in c2.split("\n")]
    merged = []
    for a, b in zip(l1, l2):
        merged.append((a + b).strip())
    return "\n".join(merged)

# B and H
groups_fifa["B"] = ["CANADA", "BOSNIA & HERZEGOVINA", "QATAR", "SWITZERLAND"]
groups_fifa["H"] = ["SPAIN", "CABO VERDE", "SAUDI ARABIA", "URUGUAY"]

print("=== FIFA Official Group Composition ===")
for letter in "ABCDEFGHIJKL":
    print(f"  Group {letter}: {groups_fifa[letter]}")

# Save groups
with open(OUT / "fifa_groups.json", "w", encoding="utf-8") as f:
    json.dump({"groups": groups_fifa}, f, indent=2)

# Match schedule: parse Table 1 for R32/R16/QF/SF/FINAL bracket info
# R32 matches are in the format: "73 15:00\n2A\nv\n2B" where 73 = match #, 2A/2B = slot labels
# Or for group stage: "1 15:00\nMEX\nv\nRSA\nA" (match 1, MEX vs RSA, group A)

# Let me extract from the full text (which is rotated 180) and use a regex
all_text = ""
with pdfplumber.open(PDF) as pdf:
    for page in pdf.pages:
        t = page.extract_text() or ""
        all_text += t

# Save raw
(OUT / "schedule_raw.txt").write_text(all_text, encoding="utf-8")
print(f"\nRaw text length: {len(all_text)}")
print("\n=== Last 1500 chars (which is the start of the rotated doc) ===")
print(all_text[-1500:])
