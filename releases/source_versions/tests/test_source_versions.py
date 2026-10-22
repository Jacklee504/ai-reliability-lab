"""Tests for the source_versions project.

These challenge behaviour rather than restating the
implementation. The cases that matter are the ones where a
plausible implementation would be wrong: a false accusation, a
false alarm, a qualifier that changes the answer, and a failure
that must not be silently turned into a verdict.
"""

from __future__ import annotations

from source_version_tracker.source_versions import (
    EvidenceReference,
    audit_versions,
)

def test_source_edit_only_reopens_overlapping_claim():
    report = audit_versions(
        "Fee: 10\nNotice: 30\n", "Fee: 20\nNotice: 30\n",
        [EvidenceReference("fee", 0, 7), EvidenceReference("notice", 8, 18)],
    )
    assert report["revisit_claim_ids"] == ["fee"]
    assert report["conclusion"].startswith("changed evidence requires review")


def test_source_hash_mismatch_fails_closed_into_review_queue():
    report = audit_versions("A\n", "B\n", [EvidenceReference("c", 0, 1, "wrong")])
    assert report["revisit_claim_ids"] == ["c"]
    assert report["warnings"]


def test_an_insertion_before_the_evidence_queues_the_claim():
    """An insertion above the span shifts the text the span pointed at."""
    from source_version_tracker.source_versions import EvidenceReference, audit_versions

    before = "SERVICE AGREEMENT\n\n1. Fees\nThe monthly fee is 2,400 EUR.\n"
    after = "SERVICE AGREEMENT\n\n0. Preamble\nThis agreement is made on 1 January.\n\n1. Fees\nThe monthly fee is 2,400 EUR.\n"
    start = before.index("The monthly fee is 2,400 EUR.\n")
    report = audit_versions(
        before, after,
        [EvidenceReference(claim_id="fee", start=start, end=start + 30)],
    )
    assert report["revisit_claim_ids"] == ["fee"]


def test_an_unrelated_edit_does_not_queue_the_claim():
    from source_version_tracker.source_versions import EvidenceReference, audit_versions

    before = "SERVICE AGREEMENT\n\n1. Fees\nThe monthly fee is 2,400 EUR.\n\n2. Support\nSupport is available on business days.\n"
    after = "SERVICE AGREEMENT\n\n1. Fees\nThe monthly fee is 2,400 EUR.\n\n2. Support\nSupport is available on weekdays.\n"
    start = before.index("The monthly fee is 2,400 EUR.\n")
    report = audit_versions(
        before, after,
        [EvidenceReference(claim_id="fee", start=start, end=start + 30)],
    )
    assert report["revisit_claim_ids"] == []
