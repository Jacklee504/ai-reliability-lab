"""Verdict construction and the rules that keep accusations honest.

The rules are applied here rather than at each call site, so that no
module can accidentally construct an unjustified contradiction.

1. A verdict may only cite evidence that was actually retrieved.
2. A universal claim ("all", "every", "only") is never contradicted by
   a passage that merely mentions the topic, and is never supported by
   a passage that describes one instance.
3. A verdict whose evidence set is empty cannot be "contradicted".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import Evidence, Method, Verdict, VerdictRecord

SUPPORTING = {Verdict.SUPPORTED}
REFUTING = {Verdict.CONTRADICTED}
ABSTAINING = {Verdict.INSUFFICIENT, Verdict.UNCERTAIN}


def make_verdict(
    claim_id: str,
    verdict: Verdict,
    *,
    explanation: str = "",
    evidence_ids: tuple[str, ...] = (),
    method: Method = Method.DETERMINISTIC,
    model_id: str | None = None,
    prompt_version: str | None = None,
    key_phrase: str | None = None,
    raw_label: str | None = None,
    parse_error: str | None = None,
    latency_ms: float | None = None,
) -> VerdictRecord:
    """Build a verdict record with the abstention rules applied."""
    if verdict is Verdict.CONTRADICTED and not evidence_ids:
        verdict = Verdict.INSUFFICIENT
        explanation = (
            (explanation + " ").strip()
            + "No retrieved evidence was available, so a contradiction cannot be asserted."
        ).strip()
    return VerdictRecord(
        claim_id=claim_id,
        verdict=verdict,
        explanation=explanation,
        evidence_ids=evidence_ids,
        method=method,
        model_id=model_id,
        prompt_version=prompt_version,
        key_phrase=key_phrase,
        raw_label=raw_label,
        parse_error=parse_error,
        latency_ms=latency_ms,
    )


def apply_abstention_rules(
    verdict: Verdict,
    claim_is_universal: bool,
    evidence_text: str,
    claim_text: str,
) -> tuple[Verdict, str]:
    """Downgrade verdicts that would be false accusations.

    Returns the (possibly adjusted) verdict and a note explaining any
    adjustment.
    """
    if verdict is Verdict.CONTRADICTED and not evidence_text.strip():
        return Verdict.INSUFFICIENT, "contradiction claimed without evidence"

    if verdict is Verdict.SUPPORTED and claim_is_universal:
        if not re.search(r"\b(?:all|every|each|any|never|no|only)\b", evidence_text, re.IGNORECASE):
            return (
                Verdict.UNCERTAIN,
                "universal claim cannot be supported by a passage that states no universality",
            )

    if verdict is Verdict.CONTRADICTED and claim_is_universal:
        if not re.search(r"\b(?:all|every|each|never|no|none|only|except)\b", evidence_text, re.IGNORECASE):
            return (
                Verdict.UNCERTAIN,
                "universal claim not refuted: the passage does not address the universal scope",
            )

    return verdict, ""


@dataclass
class VerdictTally:
    """Counts of the outcomes across a set of verdicts."""

    counts: dict[str, int]
    total: int

    @property
    def accusations(self) -> int:
        return self.counts.get(Verdict.CONTRADICTED.value, 0)

    @property
    def abstentions(self) -> int:
        return self.counts.get(Verdict.INSUFFICIENT.value, 0) + self.counts.get(
            Verdict.UNCERTAIN.value, 0
        )


def verdict_counts(verdicts: list[VerdictRecord]) -> VerdictTally:
    counts = {v.value: 0 for v in Verdict}
    for record in verdicts:
        counts[record.verdict.value] += 1
    return VerdictTally(counts=counts, total=len(verdicts))


def false_accusation_rate(
    supported_claims: list[VerdictRecord],
) -> tuple[int, int]:
    """Count (false accusations, total supported claims).

    A false accusation is a claim the source supports that was
    reported as contradicted. This is the number that most determines
    whether a reliability tool is worth trusting.
    """
    if not supported_claims:
        return 0, 0
    false_accusations = sum(1 for v in supported_claims if v.verdict is Verdict.CONTRADICTED)
    return false_accusations, len(supported_claims)


def evidence_span_map(evidence: list[Evidence]) -> dict[str, Evidence]:
    return {e.evidence_id: e for e in evidence}
