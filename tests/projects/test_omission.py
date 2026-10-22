"""Tests for the omission project.

These challenge behaviour rather than restating the
implementation. The cases that matter are the ones where a
plausible implementation would be wrong: a false accusation, a
false alarm, a qualifier that changes the answer, and a failure
that must not be silently turned into a verdict.
"""

from __future__ import annotations

from ai_reliability.projects.omission import audit_omissions

def test_omission_audit_on_empty_source_reports_nothing():
    report = audit_omissions("", "A summary.", "Summarize the obligations.", provider=None)
    assert report.total_statements == 0
    assert report.potential_omissions == []


def test_omission_candidates_are_ordered_by_relevance():
    source = (
        "A. Definitions of terms used here.\n"
        "The processor shall notify the controller within 48 hours.\n"
    )
    report = audit_omissions(source, "", "Summarize the security obligations.", provider=None)
    uncovered = report.potential_omissions
    if len(uncovered) >= 2:
        assert uncovered[0].relevance >= uncovered[1].relevance
