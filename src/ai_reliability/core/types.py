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
