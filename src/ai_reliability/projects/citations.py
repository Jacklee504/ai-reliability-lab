"""Project 3: citation support auditor.

Problem
-------
A real, relevant-looking citation may not support the sentence attached to
it. The passage shares the topic, uses the same words, and may even be
about the right entity, but it does not state the fact the sentence needs.

Approach
--------
The auditor separates two questions that are easy to conflate:

* Is the citation **topically relevant** to the claim? (cheap: token overlap)
* Does the citation **support** the claim? (expensive: entailment)

The hard negatives in this project are all cases where the first answer is
yes and the second is no. The module reports them separately so a user can
see *which* citations are merely on-topic.

Claims that genuinely need two passages together are supported too: the
auditor checks the combined evidence, not only the single best passage.

Deliberate limits
-----------------
* A citation is judged only against the passage it is attached to. The
  auditor does not go looking for a better passage elsewhere, because
  that would answer a different question rather than the one asked.
* Topical relevance is reported as a score, never as a verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.providers import (
    ModelResult,
    ParseError,
    Provider,
    ProviderError,
    parse_json_block,
)
from ..core.retrieval import lexical_support_candidate, topical_overlap
from ..core.types import (
    CaseResult,
    EvalCase,
    Evidence,
    Method,
    Span,
    Verdict,
    VerdictRecord,
    short_id,
)
from ..core.verdicts import apply_abstention_rules, make_verdict

PROMPT_VERSION = "citations-v1"

VALID_VERDICTS = ("supported", "contradicted", "insufficient_evidence", "uncertain")


@dataclass
class Citation:
    """A claim and the passages that were cited for it."""

    claim: str
    passages: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


@dataclass
class CitationFinding:
    """The audit result for one citation."""

    citation: Citation
    verdict: VerdictRecord
    topical_scores: list[float] = field(default_factory=list)
    key_phrase: str | None = None
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def max_topical(self) -> float:
        return max(self.topical_scores) if self.topical_scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation": {
                "claim": self.citation.claim,
                "passage_count": len(self.citation.passages),
            },
            "verdict": self.verdict.to_dict(),
            "topical_scores": [round(s, 4) for s in self.topical_scores],
            "max_topical_overlap": round(self.max_topical, 4),
            "key_phrase": self.key_phrase,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass
class CitationReport:
    """The full result of auditing citations."""

    findings: list[CitationFinding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    model_id: str | None = None
    model_calls: int = 0
    parse_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_calls": self.model_calls,
            "parse_failures": self.parse_failures,
            "findings": [f.to_dict() for f in self.findings],
            "notes": self.notes,
        }

    def render(self) -> str:
        lines = [f"Citation audit: {len(self.findings)} citation(s)"]
        for finding in self.findings:
            lines.append(f"  [{finding.verdict.verdict.value}] {finding.citation.claim}")
            lines.append(f"      topical overlap: {finding.max_topical:.2f}")
            if finding.verdict.explanation:
                lines.append(f"      {finding.verdict.explanation}")
            if finding.key_phrase:
                lines.append(f"      key phrase: {finding.key_phrase}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def build_citation_prompt(claim: str, passages: list[str]) -> str:
    """Build the support-check prompt for one citation.

    The prompt states the distinction the project exists to test: a
    passage can be perfectly on topic and still not support the claim.
    """
    blocks = []
    for index, passage in enumerate(passages, start=1):
        blocks.append(f"[{index}] {passage}")
    evidence_text = "\n\n".join(blocks) if blocks else "(no passage cited)"
    return (
        "You are checking whether the cited passage or passages support a claim.\n"
        "\n"
        "Rules:\n"
        "- supported: the cited passage(s) state the claim. Two passages may "
        "together support a claim that neither supports alone.\n"
        "- contradicted: the cited passage(s) state something incompatible "
        "with the claim.\n"
        "- insufficient_evidence: the cited passage(s) are on topic but do not "
        "state the fact the claim needs.\n"
        "- uncertain: you cannot decide.\n"
        "\n"
        "A passage may be perfectly relevant to the topic and still fail to "
        "support the specific claim. That is insufficient_evidence, not "
        "supported, and not contradicted.\n"
        "\n"
        f"CLAIM:\n{claim}\n"
        "\n"
        f"CITED PASSAGES:\n{evidence_text}\n"
        "\n"
        "Reply with JSON only, in this exact shape:\n"
        '{"verdict": "supported|contradicted|insufficient_evidence|uncertain", '
        '"explanation": "one sentence", "key_phrase": "the exact words in the '
        'cited passage that matter"}\n'
    )


def _classify(provider: Provider, claim: str, passages: list[str]):
    """Ask the model whether the cited passages support the claim."""
    prompt = build_citation_prompt(claim, passages)
    try:
        result: ModelResult = provider.generate(
            prompt,
            purpose="citation_support",
            prompt_version=PROMPT_VERSION,
            temperature=0.0,
            json_mode=True,
        )
    except ProviderError as exc:
        return None, f"provider error: {exc}", 0.0

    try:
        parsed = parse_json_block(result.text, raw=result.text)
    except ParseError as exc:
        return None, f"parse failure: {exc}", result.call.latency_ms

    if not isinstance(parsed, dict):
        return None, "model did not return an object", result.call.latency_ms
    return parsed, None, result.call.latency_ms


def _deterministic_verdict(claim: str, passages: list[str]) -> tuple[Verdict, str]:
    """A conservative fallback used when the model is unavailable.

    It never claims contradiction without a model, because topical
    similarity plus a number mismatch is not enough to assert that a
    passage contradicts a claim.
    """
    if not passages:
        return Verdict.INSUFFICIENT, "no passage was cited"

    best = max(passages, key=lambda p: topical_overlap(claim, p))
    overlap = topical_overlap(claim, best)
    if overlap >= 0.6 and lexical_support_candidate(claim, "\n".join(passages)):
        return Verdict.SUPPORTED, "the cited passages contain all claim terms and logical qualifiers"
    return Verdict.INSUFFICIENT, (
        f"cited passage overlap was {overlap:.0%}; the citation is on topic but "
        "was not judged to support the claim"
    )


#: The subject of a claim is the noun phrase before its verb, for
#: example "the early-career programme" in "The early-career programme
#: runs for 12 months". Two texts that talk about the same subject share
#: this phrase; two that talk about different subjects do not.
_SUBJECT_PHRASE_RE = re.compile(
    r"\b(?:the|a|an)\s+([\w-]+(?:\s+[\w-]+){0,2}?)\s+"
    r"(?:runs?|is|was|are|were|has|have|had|costs?|takes?|requires?|"
    r"includes?|contains?|states?|says?|remains?|stays?|ends?|starts?|"
    r"launches?|supports?|costs?|covers?|applies?|applies)\b",
    re.IGNORECASE,
)

#: Words that carry no entity information, so a difference in them is
#: not a difference of subject.
_STOP_SUBJECT_WORDS = {
    "programme", "program", "service", "plan", "fee", "price", "cost",
    "price", "policy", "agreement", "contract", "report", "document",
    "term", "period", "deadline", "notice", "payment", "charge",
}


def _subject_phrases(text: str) -> list[str]:
    """Extract the subject noun phrases of a text."""
    phrases: list[str] = []
    for match in _SUBJECT_PHRASE_RE.finditer(text):
        phrase = match.group(1).strip().lower()
        content = [w for w in phrase.split() if w not in _STOP_SUBJECT_WORDS]
        if content:
            phrases.append(" ".join(content))
    return phrases
