"""Shared record types for the AI reliability lab.

Every record carries enough provenance to reproduce a finding: stable IDs,
content hashes where applicable, and an explicit note when a judgement is
deterministic versus model-assisted.

Design rules that hold across the whole repository:

* A ``Span`` is the atomic unit of quoted evidence. Reports quote spans
  verbatim, never a paraphrase.
* A ``Verdict`` is one of four outcomes. "Insufficient evidence" and
  "uncertain" are distinct: the first means retrieval or the source did
  not settle the question, the second means the judge could not decide.
  A missing passage is never silently upgraded to "contradicted".
* A ``LabelOrigin`` records where an expected label came from so that
  synthetic or model-drafted labels are never reported as human ground truth.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def sha256_text(text: str) -> str:
    """Stable content hash used by source records and receipts."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def short_id(prefix: str, *parts: str) -> str:
    """Deterministic short identifier, e.g. ``sp_3f9a1c``."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"


class Verdict(str, Enum):
    """Outcome of checking a claim against supplied sources."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient_evidence"
    UNCERTAIN = "uncertain"

    @property
    def is_accusation(self) -> bool:
        """True only for verdicts that assert the answer is wrong."""
        return self is Verdict.CONTRADICTED


class Method(str, Enum):
    """How a verdict was produced."""

    DETERMINISTIC = "deterministic"
    LEXICAL = "lexical"
    MODEL = "model"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    MANUAL = "manual"
    EXECUTED = "executed"


class LabelOrigin(str, Enum):
    """Provenance of an expected label in an evaluation case."""

    HUMAN_VERIFIED = "human_verified"
    SYNTHETIC_TRANSFORM = "synthetic_transform"
    MODEL_DRAFTED = "model_drafted"
    UNKNOWN = "unknown"

    @property
    def is_human(self) -> bool:
        return self is LabelOrigin.HUMAN_VERIFIED


class Split(str, Enum):
    """Dataset split. ``TRAIN`` must never be tuned on and reported as test."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"

    @property
    def is_held_out(self) -> bool:
        return self is Split.TEST


class MediaType(str, Enum):
    TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    CSV = "text/csv"
    JSON = "application/json"
    PDF = "application/pdf"
    CODE = "text/x-python"
    UNKNOWN = "application/octet-stream"


class ExtractionStatus(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    EMPTY = "empty"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class CheckStatus(str, Enum):
    """Outcome of a single requirement or predicate check."""

    PASSED = "passed"
    FAILED = "failed"
    CANNOT_VERIFY = "cannot_verify"


@dataclass(frozen=True)
class Source:
    """A document that was supplied to, or extracted for, the audit."""

    source_id: str
    path: str | None = None
    identifier: str | None = None
    content_hash: str = ""
    version: str | None = None
    captured_at: str | None = None
    media_type: MediaType = MediaType.UNKNOWN
    extraction_status: ExtractionStatus = ExtractionStatus.OK
    warnings: tuple[str, ...] = ()
    char_count: int = 0
    page_count: int = 0
    page_marks: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "path": self.path,
            "identifier": self.identifier,
            "content_hash": self.content_hash,
            "version": self.version,
            "captured_at": self.captured_at,
            "media_type": self.media_type.value,
            "extraction_status": self.extraction_status.value,
            "warnings": list(self.warnings),
            "char_count": self.char_count,
            "page_count": self.page_count,
        }


@dataclass(frozen=True)
class Span:
    """An exact slice of extracted text from a single source.

    ``start``/``end`` are character offsets into the source's extracted text.
    ``text`` must equal ``extracted_text[start:end]`` exactly.
    """

    span_id: str
    source_id: str
    text: str
    start: int = 0
    end: int = 0
    page: int | None = None
    section: str | None = None
    warning: str | None = None
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
            "page": self.page,
            "section": self.section,
            "warning": self.warning,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class Qualifiers:
    """Atomic claim qualifiers that must survive extraction.

    Losing a qualifier is the single most common way a claim checker
    produces a false accusation, so qualifiers are carried explicitly
    rather than left implicit in the claim string.
    """

    date: str | None = None
    quantity: str | None = None
    subject: str | None = None
    scope: str | None = None
    negation: bool = False
    modality: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "quantity": self.quantity,
            "subject": self.subject,
            "scope": self.scope,
            "negation": self.negation,
            "modality": self.modality,
        }


@dataclass
class Claim:
    """An atomic, checkable assertion extracted from an answer."""

    claim_id: str
    text: str
    answer_span: str = ""
    answer_start: int = 0
    answer_end: int = 0
    qualifiers: Qualifiers = field(default_factory=Qualifiers)
    atomic: bool = True
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "answer_span": self.answer_span,
            "answer_start": self.answer_start,
            "answer_end": self.answer_end,
            "qualifiers": self.qualifiers.to_dict(),
            "atomic": self.atomic,
            "notes": self.notes,
        }
