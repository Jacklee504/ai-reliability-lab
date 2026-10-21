"""Tests for the claim auditor.

The cases that matter are the ones where a plausible implementation
would be wrong: a false accusation, a qualifier that changes the
answer, and a failure that must not be silently turned into a verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from claim_auditor.auditor import SourceDocument, audit_claims, build_claim_prompt, entity_mismatch
from claim_auditor.claims import extract_qualifiers, split_answer_into_claims
from claim_auditor.evaluation import ClaimRunner, EvalCase, load_cases, run_evaluation
from claim_auditor.providers import FakeProvider, ParseError, parse_json_block
from claim_auditor.retrieval import retrieve, topical_overlap
from claim_auditor.verdicts import apply_abstention_rules, make_verdict
from claim_auditor.types import Method, Verdict


# ---------------------------------------------------------------------------
# claim extraction
# ---------------------------------------------------------------------------


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


def test_a_coordinated_sentence_becomes_two_claims():
    answer = "The fee is 2,400 EUR and the notice period is 90 days."
    claims = split_answer_into_claims(answer)
    assert len(claims) >= 2


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


def test_cannot_contradict_without_evidence():
    record = make_verdict("c1", Verdict.CONTRADICTED, explanation="wrong")
    assert record.verdict is Verdict.INSUFFICIENT
    assert record.explanation


def test_universal_claim_is_not_contradicted_by_a_single_instance():
    verdict, note = apply_abstention_rules(
        Verdict.CONTRADICTED, True, "One customer received a refund.", "All customers receive a refund."
    )
    assert verdict is Verdict.UNCERTAIN
    assert note


# ---------------------------------------------------------------------------
# the auditor
# ---------------------------------------------------------------------------


def test_audit_never_contradicts_without_evidence():
    report = audit_claims(
        "The office is located in Dublin.",
        [SourceDocument(identifier="s", text="The monthly service fee is 2,400 EUR.")],
        provider=None,
    )
    assert report.findings
    for finding in report.findings:
        assert finding.verdict.verdict.value != "contradicted"


def test_audit_splits_a_multi_claim_sentence():
    report = audit_claims(
        "The fee is 2,400 EUR and the notice period is 60 days.",
        [SourceDocument(identifier="s", text="The fee is 2,400 EUR. The notice period is 90 days.")],
        provider=None,
    )
    assert len(report.findings) >= 2


def test_prompt_names_all_four_outcomes():
    prompt = build_claim_prompt("A claim", ["A passage"])
    for verdict in ("supported", "contradicted", "insufficient_evidence", "uncertain"):
        assert verdict in prompt


def test_a_model_parse_failure_is_recorded_not_silently_used():
    provider = FakeProvider(responses={"claim_verdict": "this is not json at all"})
    report = audit_claims(
        "The fee is 2,400 EUR.",
        [SourceDocument(identifier="s", text="The fee is 2,400 EUR.")],
        provider=provider,
    )
    assert report.parse_failures >= 1
    for finding in report.findings:
        assert finding.verdict.verdict.value != "contradicted"


def test_a_citation_about_a_different_subject_cannot_support_the_claim():
    provider = FakeProvider(responses={"claim_verdict": json.dumps({
        "verdict": "supported",
        "explanation": "The graduate programme is an early-career programme.",
        "key_phrase": "The graduate programme runs for 12 months.",
    })})
    report = audit_claims(
        "The early-career programme runs for 12 months.",
        [SourceDocument(identifier="s", text="The graduate programme runs for 12 months.")],
        provider=provider,
    )
    assert report.findings[0].verdict.verdict.value == "insufficient_evidence"


def test_entity_mismatch_is_conservative():
    assert entity_mismatch(
        "The early-career programme runs for 12 months.",
        ["The graduate programme runs for 12 months."],
    )
    assert not entity_mismatch(
        "The early-career programme runs for 12 months.",
        ["The early-career programme runs for 12 months."],
    )
    assert not entity_mismatch("The fee is 2,400 EUR.", ["The fee is described here."])


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


def test_retrieval_prefers_the_passage_that_shares_content():
    from claim_auditor.types import Span

    def span(span_id, text):
        return Span(span_id=span_id, source_id="s", text=text, start=0, end=len(text))

    spans = [
        span("s1", "The notice period is 90 days written notice."),
        span("s2", "The service fee is 2,400 EUR per month."),
    ]
    evidence = retrieve("How long is the notice period?", spans, top_k=1)
    assert evidence
    assert "notice" in evidence[0].spans[0].text


def test_topical_overlap_is_high_for_relevant_but_insufficient():
    claim = "The service fee is 2,400 EUR per calendar month."
    passage = "The service fee is described in the agreement."
    assert topical_overlap(claim, passage) > 0.25


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def test_the_fixture_set_loads_and_scores():
    cases = load_cases(ROOT / "fixtures" / "cases.json")
    assert cases
    report = run_evaluation(cases, ClaimRunner(provider=None), method="deterministic")
    assert report.total == len(cases)
    assert report.scored > 0
    assert report.metrics


def test_the_report_says_when_labels_are_not_human_verified():
    cases = load_cases(ROOT / "fixtures" / "cases.json")
    report = run_evaluation(cases, ClaimRunner(provider=None))
    assert any("not independently human verified" in note for note in report.notes)


def test_parse_json_block_tolerates_a_fenced_block():
    assert parse_json_block('```json\n{"verdict": "supported"}\n```') == {"verdict": "supported"}
