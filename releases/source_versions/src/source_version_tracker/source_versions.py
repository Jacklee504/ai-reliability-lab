"""Track source edits that invalidate an earlier evidence span.

This module queues claims for review when their cited offsets intersect a
changed region. It does not reverse an earlier verdict automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .core.types import CaseResult, EvalCase, sha256_text


@dataclass(frozen=True)
class EvidenceReference:
    claim_id: str
    start: int
    end: int
    source_hash: str | None = None


def changed_regions(before: str, after: str) -> list[dict[str, Any]]:
    """Return changed character ranges in both source versions."""
    old_lines = before.splitlines(keepends=True)
    new_lines = after.splitlines(keepends=True)
    old_offsets = [0]
    new_offsets = [0]
    for line in old_lines:
        old_offsets.append(old_offsets[-1] + len(line))
    for line in new_lines:
        new_offsets.append(new_offsets[-1] + len(line))
    regions = []
    for tag, old_a, old_b, new_a, new_b in SequenceMatcher(
        None, old_lines, new_lines, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        regions.append({
            "change": tag,
            "old_start": old_offsets[old_a], "old_end": old_offsets[old_b],
            "new_start": new_offsets[new_a], "new_end": new_offsets[new_b],
            "old_text": before[old_offsets[old_a]:old_offsets[old_b]],
            "new_text": after[new_offsets[new_a]:new_offsets[new_b]],
        })
    return regions


def audit_versions(
    before: str, after: str, references: list[EvidenceReference]
) -> dict[str, Any]:
    """List claims whose old evidence overlaps a source edit."""
    before_hash = sha256_text(before)
    after_hash = sha256_text(after)
    regions = changed_regions(before, after)
    review_ids: set[str] = set()
    warnings: list[str] = []
    for ref in references:
        if ref.start < 0 or ref.end < ref.start or ref.end > len(before):
            warnings.append(f"{ref.claim_id}: evidence offsets are outside the old source")
            review_ids.add(ref.claim_id)
            continue
        if ref.source_hash and ref.source_hash != before_hash:
            warnings.append(f"{ref.claim_id}: evidence hash does not match the old source")
            review_ids.add(ref.claim_id)
            continue
        for region in regions:
            start, end = region["old_start"], region["old_end"]
            if (start == end and ref.start <= start <= ref.end) or (
                start < ref.end and ref.start < end
            ):
                review_ids.add(ref.claim_id)
                break
            # An insertion *before* the evidence shifts the text the
            # span pointed at, so the same offsets now cover different
            # words. The claim was checked against the old text and
            # must be checked again.
            if start == end and start < ref.start:
                shifted = before[:start] + region["new_text"] + before[start:]
                if shifted[ref.start : ref.end] != before[ref.start : ref.end]:
                    review_ids.add(ref.claim_id)
                    break
    return {
        "before_hash": before_hash, "after_hash": after_hash,
        "changed_regions": regions, "revisit_claim_ids": sorted(review_ids),
        "warnings": warnings,
        "conclusion": "changed evidence requires review; no verdict was changed",
    }


@dataclass
class SourceVersionRunner:
    provider: Any = None

    def evaluate_case(self, case: EvalCase) -> CaseResult:
        data = case.input
        refs = [EvidenceReference(**item) for item in data.get("evidence", [])]
        report = audit_versions(data.get("before", ""), data.get("after", ""), refs)
        expected = case.expected
        correct = (
            report["revisit_claim_ids"] == expected["revisit_claim_ids"]
            if "revisit_claim_ids" in expected else None
        )
        return CaseResult(
            case_id=case.case_id, project=case.project, predicted=report,
            expected=expected, label_origin=case.label_origin, split=case.split,
            correct=correct,
        )


def build_cases() -> list[EvalCase]:
    from .core.fixtures import load_project_cases
    return load_project_cases("source_versions")


def build_runner(provider: Any = None) -> SourceVersionRunner:
    return SourceVersionRunner(provider=provider)
