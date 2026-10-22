"""Tests that are about the lab rather than about one project.

The per project tests live in ``tests/projects``. What remains
here is the machinery that spans the whole lab: the blind review
comparison, which every project's fixtures feed into.
"""

from __future__ import annotations

from ai_reliability.cli import compare_review_packet

def test_review_comparison_reports_disagreement_without_changing_fixture():
    result = compare_review_packet({
        "cases": [{"case_id": "claim_001_supported", "reviewed_expected": {"verdict": "contradicted"}}]
    })
    assert result["reviewed_cases"] == 1
    assert result["disagreements"][0]["case_id"] == "claim_001_supported"
    assert result["unreviewed_cases"] == result["total_cases"] - 1


def test_review_comparison_accepts_verdict_only_without_requiring_fixture_metadata():
    result = compare_review_packet({
        "cases": [{"case_id": "claim_001_supported", "reviewed_expected": {"verdict": "supported"}}]
    })
    assert result["reviewed_cases"] == 1
    assert result["disagreements"] == []


def test_review_comparison_surfaces_draft_alternatives_even_when_choice_matches():
    result = compare_review_packet({
        "cases": [{"case_id": "claim_004_offtopic_source", "reviewed_expected": {"verdict": "insufficient_evidence"}}]
    })
    assert result["disagreements"] == []
    assert result["ambiguous_drafts"] == [{
        "case_id": "claim_004_offtopic_source",
        "field": "verdict",
        "draft_options": ["insufficient_evidence", "uncertain"],
    }]
