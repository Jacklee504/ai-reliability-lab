"""Tests for the claims project.

These challenge behaviour rather than restating the
implementation. The cases that matter are the ones where a
plausible implementation would be wrong: a false accusation, a
false alarm, a qualifier that changes the answer, and a failure
that must not be silently turned into a verdict.
"""

from __future__ import annotations

from claim_auditor.core.providers import FakeProvider, ParseError
from claim_auditor.core.claims import split_answer_into_claims
from claim_auditor.claims import (
    SourceDocument,
    audit_claims,
    build_claim_prompt,
)

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


def test_a_coordinated_sentence_becomes_two_claims():
    """One sentence with two coordinated clauses carries two claims."""
    answer = "The fee is 2,400 EUR and the notice period is 90 days."
    claims = split_answer_into_claims(answer)
    assert len(claims) >= 2
    for claim in claims:
        assert answer[claim.answer_start : claim.answer_end] == claim.text


def test_the_fixture_set_loads_and_scores():
    """The shipped cases must load and every one must be scored."""
    from claim_auditor.core.fixtures import load_cases
    from claim_auditor.claims import build_runner

    cases = load_cases("claims")
    assert cases
    runner = build_runner(provider=None)
    for case in cases:
        result = runner.evaluate_case(case)
        assert result.predicted
        assert result.correct is not None, case.case_id


def test_the_report_says_when_the_labels_are_not_human_verified():
    """A report that does not say this is quoting numbers it should not."""
    from claim_auditor.core.evaluation import run_evaluation
    from claim_auditor.core.fixtures import load_cases
    from claim_auditor.claims import build_runner

    report = run_evaluation(
        "claims", load_cases("claims"), build_runner(provider=None)
    )
    assert any("not independently human verified" in n for n in report.notes)
