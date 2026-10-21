"""Generate the synthetic PDF fixture used by the receipt tests.

The fixture is generated rather than committed as a binary so that its
contents are reviewable in text form. Run it with:

    python tests/make_pdf_fixture.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

TARGET = Path(__file__).resolve().parent.parent / "fixtures" / "context_receipts" / "sample_policy.pdf"

PAGES = [
    [
        "DATA PROCESSING ADDENDUM",
        "",
        "1. Scope and Definitions",
        "Personal data means any information relating to an identified",
        "or identifiable natural person.",
        "The controller is the entity that determines the purposes of",
        "processing.",
    ],
    [
        "2. Security Measures",
        "",
        "The processor shall implement appropriate technical and",
        "organisational measures.",
        "The processor shall notify the controller of any personal data",
        "breach within 48 hours.",
    ],
    [
        "3. International Transfers",
        "",
        "Personal data shall not be transferred outside the European",
        "Economic Area.",
        "Transfers outside the European Economic Area require",
        "appropriate safeguards.",
    ],
]


def build(path: Path = TARGET) -> Path:
    """Write a three-page PDF with known text on every page."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    for lines in PAGES:
        y = height - 72
        for line in lines:
            pdf.drawString(72, y, line)
            y -= 18
        pdf.showPage()
    pdf.save()
    return path


if __name__ == "__main__":
    print("wrote " + str(build()))
