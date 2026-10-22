"""Tests for the study_coverage project.

These challenge behaviour rather than restating the
implementation. The cases that matter are the ones where a
plausible implementation would be wrong: a false accusation, a
false alarm, a qualifier that changes the answer, and a failure
that must not be silently turned into a verdict.
"""

from __future__ import annotations

from study_coverage.study_coverage import audit_coverage

def test_study_paraphrase_does_not_become_an_omission_claim():
    report = audit_coverage(
        [{"id": "recursion", "label": "recursion"}],
        "A function calls itself to solve smaller problems.",
    )
    assert report["topics"][0]["status"] == "unverified"


def test_study_counts_label_and_alias_mentions_without_double_counting():
    report = audit_coverage(
        [{"id": "pcr", "label": "polymerase chain reaction", "aliases": ["PCR"]}],
        "PCR is short for polymerase chain reaction. PCR is common.",
    )
    assert report["topics"][0]["mentions"] == 3
