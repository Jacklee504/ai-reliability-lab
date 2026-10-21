"""Tests for the project modules.

These challenge behaviour rather than restating the implementation. The
cases that matter are the ones where a plausible implementation would be
wrong: a false accusation, a false alarm, a qualifier that changes the
answer, and a failure that must not be silently turned into a verdict.
"""

from __future__ import annotations

import json

from ai_reliability.core.providers import FakeProvider, ParseError
from ai_reliability.cli import compare_review_packet
from ai_reliability.core.types import EvalCase, LabelOrigin, Split
from ai_reliability.projects.authorization import (
    Authority,
    Authorization,
    ProposedAction,
    evaluate_action,
)
from ai_reliability.projects.baselines import BaselineRunner, run_naive_baseline
from ai_reliability.projects.citations import Citation, audit_citations
from ai_reliability.projects.claims import SourceDocument, audit_claims, build_claim_prompt
from ai_reliability.projects.code_review import (
    CodeSnapshot,
    ReviewComment,
    extract_assertion,
    verify_review_comment,
    verify_runtime_claim,
)
from ai_reliability.projects.deliverables import check_deliverable, extract_requirements
from ai_reliability.projects.instructions import (
    check_compliance,
    parse_instructions,
)
from ai_reliability.projects.numerical import (
    parse_table,
    verify_numeric_claim,
)
from ai_reliability.projects.omission import audit_omissions
from ai_reliability.projects.source_versions import EvidenceReference, audit_versions
from ai_reliability.projects.study_coverage import audit_coverage


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


def test_claim_audit_never_contradicts_without_evidence():
    """A claim with no matching source must abstain, never accuse."""
    report = audit_claims(
        "The office is located in Dublin.",
        [SourceDocument(identifier="s", text="The monthly service fee is 2,400 EUR.")],
        provider=None,
    )
    assert report.findings
    for finding in report.findings:
        assert finding.verdict.verdict.value != "contradicted"


def test_claim_audit_splits_a_multi_claim_sentence():
    """One sentence with two claims must produce two findings."""
    report = audit_claims(
        "The fee is 2,400 EUR and the notice period is 60 days.",
        [SourceDocument(identifier="s", text="The fee is 2,400 EUR. The notice period is 90 days.")],
        provider=None,
    )
    assert len(report.findings) >= 2


def test_claim_prompt_names_all_four_outcomes():
    prompt = build_claim_prompt("A claim", ["A passage"])
    for verdict in ("supported", "contradicted", "insufficient_evidence", "uncertain"):
        assert verdict in prompt


def test_claim_audit_records_a_model_parse_failure():
    """A model that returns nonsense must be recorded, not silently used."""
    provider = FakeProvider(
        responses={"claim_verdict": "this is not json at all"},
    )
    report = audit_claims(
        "The fee is 2,400 EUR.",
        [SourceDocument(identifier="s", text="The fee is 2,400 EUR.")],
        provider=provider,
    )
    assert report.parse_failures >= 1
    for finding in report.findings:
        assert finding.verdict.verdict.value != "contradicted"


def test_naive_judges_keep_nonanswers_out_of_binary_verdicts():
    binary = FakeProvider(responses={"naive_binary": "Not enough information."})
    assert run_naive_baseline(binary, "A", ["B"]).verdict is None
    three_way = FakeProvider(responses={"naive_verdict": "Not enough information."})
    assert run_naive_baseline(three_way, "A", ["B"], "naive_verdict").verdict.value == "insufficient_evidence"


def test_multi_claim_baseline_requires_the_expected_sequence():
    case = EvalCase(
        case_id="mixed", project="claims",
        input={"answer": "The fee is 10 EUR and the notice is 30 days.", "sources": []},
        expected={"verdicts": ["supported", "contradicted"]},
    )
    always_yes = FakeProvider(responses={"naive_binary": "Yes"})
    result = BaselineRunner(provider=always_yes).evaluate_case(case)
    assert result.predicted["verdicts"] == ["supported", "supported"]
    assert result.correct is False
