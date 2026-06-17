"""Extract 2026 WC schedule from the official PDF and compare with the dataset."""
import sys
import re
from pathlib import Path
import pandas as pd
import pdfplumber

PDF = Path(r"C:\Users\abhir\Downloads\FWC26 Match Schedule_v17_10042026_EN.pdf")
DATA = Path(r"C:\Users\abhir\Downloads\football_analysis")

with pdfplumber.open(PDF) as pdf:
    print(f"PDF pages: {len(pdf.pages)}")
    all_text = ""
    for i, page in enumerate(pdf.pages):
        t = page.extract_text() or ""
        all_text += f"\n=== PAGE {i+1} ===\n{t}"

# Save raw text
(Path(r"C:\Users\abhir\.mavis\sessions\mvs_fbdb1a4d0ee34fc9b22f88407249a52a\workspace\football_2026") / "schedule_pdf_text.txt").write_text(all_text, encoding="utf-8")
print(f"Saved raw text ({len(all_text)} chars)")

# Print first ~3000 chars to see structure
print("\n=== FIRST 3000 CHARS ===")
print(all_text[:3000])
