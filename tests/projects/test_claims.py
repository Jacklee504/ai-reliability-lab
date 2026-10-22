"""Tests for the claims project.

These challenge behaviour rather than restating the
implementation. The cases that matter are the ones where a
plausible implementation would be wrong: a false accusation, a
false alarm, a qualifier that changes the answer, and a failure
that must not be silently turned into a verdict.
"""

from __future__ import annotations

from ai_reliability.core.providers import FakeProvider, ParseError
from ai_reliability.projects.claims import (
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
