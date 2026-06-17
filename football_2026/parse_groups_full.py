"""More thorough PDF parse: pull every match cell with date, time, code, group."""
import pdfplumber
import re
import json
from pathlib import Path

PDF = Path(r"C:\Users\abhir\Downloads\FWC26 Match Schedule_v17_10042026_EN.pdf")
OUT = Path(r"C:\Users\abhir\.mavis\sessions\mvs_fbdb1a4d0ee34fc9b22f88407249a52a\workspace\football_2026")

with pdfplumber.open(PDF) as pdf:
    page = pdf.pages[0]
    tables = page.extract_tables()

# Table 2 has all 12 groups. Let me get all the cells more carefully
group_table = tables[2]
all_cells = []
for row in group_table:
    if not row:
        continue
    for cell in row:
        if cell:
            all_cells.append(cell)

# Print all cells
for i, c in enumerate(all_cells):
    print(f"--- Cell {i} ---")
    print(c)
    print()
