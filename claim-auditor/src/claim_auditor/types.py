"""Shared record types for the claim auditor.

The types carry the provenance needed to reproduce a finding: which
source a span came from, which evidence supported a verdict, and which
model produced it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """The four outcomes the auditor may report.

    ``INSUFFICIENT`` is deliberately not a softer "contradicted". It
    means the retrieved evidence did not settle the question, and it is
    reported separately so that a retrieval miss never appears as a
    false accusation.
    """

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient_evidence"
    UNCERTAIN = "uncertain"


class Method(str, Enum):
    """How a verdict was produced."""

    DETERMINISTIC = "deterministic"
    LEXICAL = "lexical"
    MODEL = "model"
    EXECUTED = "executed"


@dataclass(frozen=True)
class Source:
    """A document the answer may be checked against."""

    source_id: str
    identifier: str = ""
    content_hash: str = ""
    version: str | None = None
    media_type: str = "text/plain"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "identifier": self.identifier,
            "content_hash": self.content_hash,
            "version": self.version,
            "media_type": self.media_type,
        }


@dataclass(frozen=True)
class Span:
    """An exact slice of a source's text.

    ``text`` must equal the slice it points at. A span that does not
    match its own offsets is a bug, and the tests check that it never
    happens.
    """

    span_id: str
    source_id: str
    text: str
    start: int = 0
    end: int = 0
    section: str | None = None
    page: int | None = None
    kind: str = "text"

    def quote(self, max_chars: int = 240) -> str:
        text = self.text.strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1] + "…"

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "source_id": self.source_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "section": self.section,
            "page": self.page,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class Qualifiers:
    """The qualifiers that must survive claim extraction.

    Losing a qualifier is the most common way a claim checker produces
    a false accusation, so they are carried explicitly.
    """

    date: str | None = None
    quantity: str | None = None
    subject: str | None = None
    scope: str | None = None
    negation: bool = False
    modality: str | None = None


@dataclass(frozen=True)
class Claim:
    """One atomic claim, with the answer span it came from."""

    claim_id: str
    text: str
    answer_start: int = 0
    answer_end: int = 0
    qualifiers: Qualifiers = field(default_factory=Qualifiers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "answer_start": self.answer_start,
            "answer_end": self.answer_end,
            "qualifiers": {
                "date": self.qualifiers.date,
                "quantity": self.qualifiers.quantity,
                "subject": self.qualifiers.subject,
                "scope": self.qualifiers.scope,
                "negation": self.qualifiers.negation,
                "modality": self.qualifiers.modality,
            },
        }


@dataclass(frozen=True)
class Evidence:
    """One retrieved passage, with its retrieval score.

    A retrieval score is *topical relevance*. It is never evidence of
    entailment.
    """

    evidence_id: str
    spans: tuple[Span, ...] = ()
    retrieval_score: float | None = None
    method: str = "lexical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "spans": [s.to_dict() for s in self.spans],
            "retrieval_score": self.retrieval_score,
            "method": self.method,
        }


@dataclass(frozen=True)
class VerdictRecord:
    """The verdict on one claim, with its provenance."""

    claim_id: str
    verdict: Verdict
    explanation: str = ""
    evidence_ids: tuple[str, ...] = ()
    method: Method = Method.DETERMINISTIC
    model_id: str | None = None
    prompt_version: str | None = None
    key_phrase: str | None = None
    raw_label: str | None = None
    parse_error: str | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "verdict": self.verdict.value,
            "explanation": self.explanation,
            "evidence_ids": list(self.evidence_ids),
            "method": self.method.value,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "key_phrase": self.key_phrase,
            "raw_label": self.raw_label,
            "parse_error": self.parse_error,
            "latency_ms": self.latency_ms,
        }


def sha256_text(text: str) -> str:
    """Stable content hash for a piece of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_id(prefix: str, *parts: str) -> str:
    """A short, stable identifier derived from its parts."""
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8"))
    return "%s_%s" % (prefix, digest.hexdigest()[:12])
