"""Tests for the shared core: types, sources, spans, claims, retrieval, verdicts.

These tests challenge behaviour rather than restating the implementation.
The ones that matter most are the failure cases: empty sources, failed
extraction, qualifiers that change a verdict, and a verdict that tries to
accuse without evidence.
"""

from __future__ import annotations

import json
import pytest

from conftest import FIXTURES

from ai_reliability.core.claims import extract_qualifiers, split_answer_into_claims
from ai_reliability.core.providers import OllamaProvider, ProviderError, load_cache
from ai_reliability.core.retrieval import retrieve, topical_overlap
from ai_reliability.core.sources import load_source
from ai_reliability.core.spans import segment_statements, split_sections
from ai_reliability.core.types import (
    ExtractionStatus,
    Method,
    Source,
    Span,
    Verdict,
    sha256_text,
)
from ai_reliability.core.verdicts import apply_abstention_rules, make_verdict


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------


def test_missing_file_fails_loudly(tmp_path):
    source, text = load_source(tmp_path / "nope.txt")
    assert source.extraction_status is ExtractionStatus.FAILED
    assert text == ""
    assert any("not found" in w for w in source.warnings)


def test_empty_file_is_reported_as_empty_not_ok(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    source, text = load_source(path)
    assert source.extraction_status is ExtractionStatus.EMPTY
    assert text == ""
    assert any("empty" in w for w in source.warnings)


def test_unknown_media_type_produces_a_warning(tmp_path):
    path = tmp_path / "thing.bin"
    path.write_text("some bytes", encoding="utf-8")
    source, _text = load_source(path)
    assert any("unknown media type" in w for w in source.warnings)


def test_content_hash_changes_with_content(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("one", encoding="utf-8")
    b.write_text("two", encoding="utf-8")
    source_a, _ = load_source(a)
    source_b, _ = load_source(b)
    assert source_a.content_hash != source_b.content_hash
    assert source_a.content_hash == sha256_text("one") or len(source_a.content_hash) == 64


# --------------------------------------------------------------------------
# PDF extraction
# --------------------------------------------------------------------------


def test_pdf_extraction_returns_the_expected_text_and_page_count():
    """A generated PDF must extract its text and report its real page count."""
    pdf_path = FIXTURES / "context_receipts" / "sample_policy.pdf"
    if not pdf_path.exists():
        pytest.skip("the generated PDF fixture is missing; run tests/make_pdf_fixture.py")
    source, text = load_source(pdf_path)
    assert source.extraction_status is ExtractionStatus.OK
    assert source.media_type.value == "application/pdf"
    assert source.page_count == 3
    assert "Personal data means any information" in text
    assert "within 48 hours" in text
    assert "European Economic Area" in text


def test_pdf_page_marks_point_at_the_start_of_each_page():
    """The recorded page offsets must actually land on page boundaries."""
    pdf_path = FIXTURES / "context_receipts" / "sample_policy.pdf"
    if not pdf_path.exists():
        pytest.skip("the generated PDF fixture is missing; run tests/make_pdf_fixture.py")
    source, text = load_source(pdf_path)
    assert len(source.page_marks) == source.page_count
    for mark in source.page_marks:
        assert 0 <= mark <= len(text)
    # Each page's own heading must appear at or after that page's mark.
    headings = ["DATA PROCESSING ADDENDUM", "2. Security Measures", "3. International Transfers"]
    for mark, heading in zip(source.page_marks, headings):
        assert text.index(heading) >= mark


def test_pdf_receipt_records_the_page_count_and_page_spanning_chunk():
    """The receipt must show the real page count, not a form-feed guess."""
    from ai_reliability.projects.context_receipts import build_file_receipt

    pdf_path = FIXTURES / "context_receipts" / "sample_policy.pdf"
    if not pdf_path.exists():
        pytest.skip("the generated PDF fixture is missing; run tests/make_pdf_fixture.py")
    receipt = build_file_receipt(pdf_path)
    assert receipt.page_count == 3
    assert receipt.source.page_count == 3
    assert receipt.warnings == []
    assert receipt.chunks
    for chunk in receipt.chunks:
        assert chunk.page is not None
        assert 1 <= chunk.page <= 3


def test_pdf_with_no_extractable_text_is_reported_as_empty(tmp_path):
    """A PDF with no text layer must not look like a successful extraction."""
    reportlab = pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    path = tmp_path / "blank.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.showPage()
    pdf.save()

    source, text = load_source(path)
    assert source.extraction_status is ExtractionStatus.EMPTY
    assert text.strip() == ""
    assert any("no extractable text" in w for w in source.warnings)
