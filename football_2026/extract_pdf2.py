"""Re-extract the rotated PDF using a per-page rotation."""
import pdfplumber
from pathlib import Path

PDF = Path(r"C:\Users\abhir\Downloads\FWC26 Match Schedule_v17_10042026_EN.pdf")
OUT = Path(r"C:\Users\abhir\.mavis\sessions\mvs_fbdb1a4d0ee34fc9b22f88407249a52a\workspace\football_2026")

all_text = ""
with pdfplumber.open(PDF) as pdf:
    for i, page in enumerate(pdf.pages):
        # Try the page as-is first; if text is mostly numeric fragments, it's rotated
        text_normal = page.extract_text() or ""
        # Use to_image + rotation approach
        # pdfplumber has a way to set rotation via the page_obj
        from pdfplumber.page import Page
        # easier: read chars and reverse their y-coordinates
        chars = page.chars
        h = page.height
        w = page.width
        # Re-project chars as if page were rotated 180 (y' = h - y, x' = w - x)
        rotated = []
        for c in chars:
            nc = dict(c)
            nc["x0"], nc["x1"] = w - c["x1"], w - c["x0"]
            nc["top"], nc["bottom"] = h - c["bottom"], h - c["top"]
            rotated.append(nc)
        # Build new page from rotated chars
        from pdfplumber.utils import extract_text
        # Use the simpler API: re-extract using a custom approach
        # Fallback: just print flipped text
        text_rotated_chars = sorted(rotated, key=lambda c: (-c["top"], c["x0"]))
        text_rotated = " ".join(c["text"] for c in text_rotated_chars)
        # Heuristic: rotated text should have more alphabetic words
        words_normal = sum(c.isalpha() for c in text_normal)
        words_rotated = sum(c.isalpha() for c in text_rotated)
        if words_rotated > words_normal:
            all_text += f"\n=== PAGE {i+1} (rotated) ===\n{text_rotated}"
        else:
            all_text += f"\n=== PAGE {i+1} ===\n{text_normal}"

(OUT / "schedule_pdf_text.txt").write_text(all_text, encoding="utf-8")
print(f"Saved {len(all_text)} chars")

# Show first 2500 chars
print("\n=== FIRST 2500 CHARS ===")
print(all_text[:2500])
