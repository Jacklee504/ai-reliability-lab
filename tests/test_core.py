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


# --------------------------------------------------------------------------
# spans
# --------------------------------------------------------------------------


def test_sections_split_on_numbered_headings():
    text = "Preamble line.\n\n1. First Part\nBody text here.\n\n2. Second Part\nMore body.\n"
    sections = split_sections(text)
    titles = [s.title for s in sections]
    assert any("First Part" in t for t in titles)
    assert any("Second Part" in t for t in titles)


def test_segments_never_lose_text():
    text = (
        "The fee is 2,400 EUR per month. Payment is due within 30 days.\n"
        "- The customer may terminate with 90 days notice.\n"
        "- Late interest of 1.5 percent applies.\n"
    )
    source = Source(source_id="src_test", path="t.txt", content_hash="h")
    statements = segment_statements(text, source)
    assert statements
    for statement in statements:
        assert text[statement.start : statement.end] == statement.text


def test_empty_source_yields_no_statements():
    source = Source(source_id="src_test", path="t.txt", content_hash="h")
    assert segment_statements("   \n  \n", source) == []


def test_bulleted_items_become_separate_statements():
    text = "- First obligation here.\n- Second obligation here.\n"
    source = Source(source_id="src_test", path="t.txt", content_hash="h")
    statements = segment_statements(text, source)
    assert len(statements) == 2
    assert statements[0].kind == "list_item"


# --------------------------------------------------------------------------
# claims and qualifiers
# --------------------------------------------------------------------------


def test_claim_extraction_preserves_offsets():
    answer = "The fee is 2,400 EUR. Payment is due within 30 days."
    claims = split_answer_into_claims(answer)
    assert len(claims) == 2
    for claim in claims:
        assert answer[claim.answer_start : claim.answer_end] == claim.text


def test_qualifier_scan_finds_date_quantity_and_scope():
    q = extract_qualifiers("All applicants must re-enrol before 2024 for 2,400 EUR.")
    assert q.date == "2024"
    assert q.quantity is not None
    assert q.scope == "all"


def test_negation_is_detected():
    q = extract_qualifiers("The customer does not pay the late interest fee.")
    assert q.negation is True


def test_qualifier_missing_versus_present_differs():
    with_all = extract_qualifiers("All applicants must re-enrol.")
    with_some = extract_qualifiers("Some applicants must re-enrol.")
    assert with_all.scope != with_some.scope


def test_split_keeps_both_sides_of_a_contrast():
    answer = "The fee is low, but the notice period is 90 days."
    claims = split_answer_into_claims(answer)
    texts = [c.text.lower() for c in claims]
    assert any("fee is low" in t for t in texts)
    assert any("notice period" in t for t in texts)


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------


def test_retrieval_prefers_the_passage_that_shares_content():
    spans = [
        _span("s1", "The notice period is 90 days written notice."),
        _span("s2", "The service fee is 2,400 EUR per month."),
    ]
    evidence = retrieve("How long is the notice period?", spans, top_k=1)
    assert evidence
    assert "notice" in evidence[0].spans[0].text


def test_retrieval_returns_nothing_for_unrelated_query():
    spans = [_span("s1", "The fee is 2,400 EUR per month.")]
    assert retrieve("What colour is the car?", spans, top_k=3) == []


def test_injected_embeddings_rank_spans_without_claiming_entailment():
    from ai_reliability.core.retrieval import retrieve_semantic

    class FakeEmbedder:
        def embed(self, texts):
            assert len(texts) == 3
            return [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]

    spans = [_span("s1", "Unrelated."), _span("s2", "Relevant but not necessarily supporting.")]
    evidence = retrieve_semantic("Claim", spans, FakeEmbedder(), top_k=1)
    assert evidence[0].spans[0].span_id == "s2"
    assert evidence[0].method == "embedding"


def test_topical_overlap_is_high_for_relevant_but_insufficient():
    claim = "The service fee is 2,400 EUR per calendar month."
    passage = "The service fee is described in the agreement."
    overlap = topical_overlap(claim, passage)
    assert overlap > 0.25


def _span(span_id: str, text: str) -> Span:
    return Span(span_id=span_id, source_id="src_x", text=text, start=0, end=len(text))

def test_hashed_embeddings_are_deterministic_and_normalised():
    from ai_reliability.core.embeddings import HashedEmbeddingProvider, hashed_embedding

    first = hashed_embedding("the fee is 2,400 EUR")
    second = hashed_embedding("the fee is 2,400 EUR")
    assert first == second
    norm = sum(value * value for value in first) ** 0.5
    assert abs(norm - 1.0) < 1e-9 or first == [0.0] * len(first)


def test_hashed_embeddings_separate_different_texts():
    from ai_reliability.core.embeddings import cosine, hashed_embedding

    a = hashed_embedding("the monthly service fee is 2,400 EUR")
    b = hashed_embedding("either party may terminate with 90 days notice")
    same = hashed_embedding("the monthly service fee is 2,400 EUR")
    assert cosine(a, same) > 0.99
    assert cosine(a, b) < 0.5


def test_embedding_provider_satisfies_the_retrieval_protocol():
    from ai_reliability.core.embeddings import HashedEmbeddingProvider
    from ai_reliability.core.retrieval import retrieve_semantic

    provider = HashedEmbeddingProvider()
    spans = [
        _span("s1", "The notice period is 90 days written notice."),
        _span("s2", "The service fee is 2,400 EUR per month."),
    ]
    evidence = retrieve_semantic("What is the notice period?", spans, provider, top_k=2)
    assert evidence
    assert all(item.method == "embedding" for item in evidence)


def test_retrieval_comparison_reports_both_rankers():
    from ai_reliability.core.fixtures import load_project_cases
    from ai_reliability.projects.retrieval_eval import evaluate_retrieval, summarise

    cases = load_project_cases("retrieval_eval")
    assert cases
    outcomes = evaluate_retrieval(cases[0], top_k=3)
    assert set(outcomes) == {"lexical", "embedding"}
    for outcome in outcomes.values():
        assert outcome.method in ("lexical", "embedding")
        assert isinstance(outcome.hit, bool)


def test_retrieval_summary_reports_recall_and_rank():
    from ai_reliability.core.fixtures import load_project_cases
    from ai_reliability.projects.retrieval_eval import RetrievalRunner, summarise

    cases = load_project_cases("retrieval_eval")
    runner = RetrievalRunner()
    results = [runner.evaluate_case(case) for case in cases]
    summary = summarise(results)
    for method in ("lexical", "embedding"):
        block = summary["methods"][method]
        assert block["cases"] == len(cases)
        assert 0 <= block["recall_at_k"] <= block["recall_denominator"]
        assert block["mean_reciprocal_rank"] is not None


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------


def test_cannot_contradict_without_evidence():
    record = make_verdict("c1", Verdict.CONTRADICTED, explanation="wrong")
    assert record.verdict is Verdict.INSUFFICIENT
    assert record.explanation


def test_universal_claim_is_not_contradicted_by_a_single_instance():
    verdict, note = apply_abstention_rules(
        Verdict.CONTRADICTED,
        claim_is_universal=True,
        evidence_text="One customer received a refund in March.",
        claim_text="All customers receive a refund.",
    )
    assert verdict is Verdict.UNCERTAIN
    assert note


def test_universal_claim_is_not_supported_by_a_single_instance():
    verdict, note = apply_abstention_rules(
        Verdict.SUPPORTED,
        claim_is_universal=True,
        evidence_text="One customer received a refund in March.",
        claim_text="All customers receive a refund.",
    )
    assert verdict is Verdict.UNCERTAIN


def test_shall_does_not_accidentally_count_as_all():
    verdict, _note = apply_abstention_rules(
        Verdict.SUPPORTED,
        claim_is_universal=True,
        evidence_text="One customer shall receive a refund.",
        claim_text="All customers receive a refund.",
    )
    assert verdict is Verdict.UNCERTAIN


def test_contradiction_without_any_evidence_becomes_insufficient():
    verdict, _note = apply_abstention_rules(
        Verdict.CONTRADICTED,
        claim_is_universal=False,
        evidence_text="",
        claim_text="The fee is 2,400 EUR.",
    )
    assert verdict is Verdict.INSUFFICIENT


def test_matching_verdict_keeps_its_method():
    record = make_verdict("c1", Verdict.SUPPORTED, method=Method.MODEL, model_id="m")
    assert record.method is Method.MODEL
    assert record.model_id == "m"


def test_missing_explicit_model_cache_fails_closed(tmp_path):
    with pytest.raises(ProviderError, match="does not exist"):
        load_cache(tmp_path / "missing.json", expected_model_id="m")


def test_model_cache_cannot_be_attributed_to_another_model(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"model": "first", "responses": {"k": "yes"}}), encoding="utf-8")
    with pytest.raises(ProviderError, match="different model"):
        load_cache(path, expected_model_id="second")


def test_configured_cache_miss_never_falls_through_to_network(tmp_path, monkeypatch):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"model": "m", "responses": {}}), encoding="utf-8")
    monkeypatch.setenv("AIRL_MODEL_CACHE", str(path))
    provider = OllamaProvider(model_id="m")
    with pytest.raises(ProviderError, match="no network call made"):
        provider.generate("not cached", purpose="test")
