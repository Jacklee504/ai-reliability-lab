"""Tests for the citations project.

These challenge behaviour rather than restating the
implementation. The cases that matter are the ones where a
plausible implementation would be wrong: a false accusation, a
false alarm, a qualifier that changes the answer, and a failure
that must not be silently turned into a verdict.
"""

from __future__ import annotations

import json

from ai_reliability.core.providers import FakeProvider
from ai_reliability.projects.citations import Citation, audit_citations

def test_citation_audit_reports_topical_overlap_separately():
    """Topical relevance must never be presented as support."""
    report = audit_citations(
        [Citation(claim="The fee is 2,400 EUR.", passages=["The fee is described here."])],
        provider=None,
    )
    finding = report.findings[0]
    assert finding.max_topical > 0.0
    assert finding.verdict.verdict.value != "supported"


def test_a_citation_about_a_different_entity_cannot_support_the_claim():
    """A model that bridges two named entities must not be believed."""
    provider = FakeProvider(responses={"citation_support": json.dumps({
        "verdict": "supported",
        "explanation": "The graduate programme is an early-career programme.",
        "key_phrase": "The graduate programme runs for 12 months.",
    })})
    report = audit_citations(
        [Citation(
            claim="The early-career programme runs for 12 months.",
            passages=["The graduate programme runs for 12 months."],
        )],
        provider=provider,
    )
    finding = report.findings[0]
    assert finding.verdict.verdict.value == "insufficient_evidence"
    assert "different entity" in finding.verdict.explanation


def test_a_citation_that_mentions_the_claimed_entity_is_not_downgraded():
    provider = FakeProvider(responses={"citation_support": json.dumps({
        "verdict": "supported",
        "explanation": "The passage states the duration.",
        "key_phrase": "runs for 12 months",
    })})
    report = audit_citations(
        [Citation(
            claim="The early-career programme runs for 12 months.",
            passages=["The early-career programme runs for 12 months."],
        )],
        provider=provider,
    )
    assert report.findings[0].verdict.verdict.value == "supported"
