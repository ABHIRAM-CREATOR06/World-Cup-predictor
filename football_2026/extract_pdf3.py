"""Use pypdfium2 to render the PDF page as image (it auto-handles rotation),
then OCR is unnecessary because pdfplumber's extract_text actually does work
on the original page — but the page is rotated 180 in the PDF metadata.

Strategy: read with pdfplumber, take chars, and manually un-rotate them.
"""
import pdfplumber
from pathlib import Path
import pandas as pd

PDF = Path(r"C:\Users\abhir\Downloads\FWC26 Match Schedule_v17_10042026_EN.pdf")
OUT = Path(r"C:\Users\abhir\.mavis\sessions\mvs_fbdb1a4d0ee34fc9b22f88407249a52a\workspace\football_2026")

# Group stage match dates: 72 group matches already in the dataset.
# The PDF has the WHOLE tournament schedule including knockout dates,
# plus kickoff times and venues. Let me extract it cleanly.

with pdfplumber.open(PDF) as pdf:
    page = pdf.pages[0]
    print(f"Page size: {page.width} x {page.height}")
    print(f"Page rotation: {getattr(page, 'rotation', 'N/A')}")
    print(f"mediabox: {page.mediabox}")

    # Use pypdfium2 to render with auto-rotation
    import pypdfium2 as pdfium
    pdf_doc = pdfium.PdfDocument(str(PDF))
    pg = pdf_doc[0]
    print(f"pypdfium2 rotation: {pg.get_rotation()}")
    pil_image = pg.render(scale=2.0).to_pil()
    pil_image.save(OUT / "schedule_page.png")
    print(f"Rendered page saved to {OUT / 'schedule_page.png'}")

    # Now look at extract_tables to see if it has structured data
    tables = page.extract_tables()
    print(f"Found {len(tables)} tables")
    for i, t in enumerate(tables[:3]):
        print(f"\n--- Table {i} ---")
        for row in t[:5]:
            print(row)
