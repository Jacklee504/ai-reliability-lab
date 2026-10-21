"""Project 5: file-context receipt.

Problem
-------
A user of a controlled AI pipeline cannot tell which file version and
which extracted text were actually submitted to the model. When the
pipeline silently extracted an older revision, or skipped a page, or sent
an empty file, the model still answers confidently and the user has no
way to notice.

Approach
--------
Build a *receipt*: a machine-readable record of exactly what our pipeline
extracted and submitted. The receipt contains content hashes, page and
section counts, extracted character counts, warnings, chunk IDs, and the
exact chunk text selected for submission. Two receipts can be diffed to
show what changed.

Deliberate limits
-----------------
The receipt proves what *our pipeline* submitted. It cannot prove what a
model used in its reasoning, and it never claims to. A third-party
application that we did not instrument is out of scope.

This module makes no model calls.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..core.fixtures import project_fixture_dir
from ..core.sources import load_source
from ..core.spans import segment_statements, split_sections
from ..core.types import (
    CaseResult,
    EvalCase,
    ExtractionStatus,
    MediaType,
    Source,
    sha256_text,
)

RECEIPT_VERSION = "1.0"

#: Characters per chunk. Chunks are statement-aligned so that a chunk
#: boundary never splits an obligation across two chunks.
DEFAULT_CHUNK_CHARS = 1200


@dataclass
class Chunk:
    """A unit of text selected for submission to the model."""

    chunk_id: str
    source_id: str
    text: str
    start: int
    end: int
    char_count: int
    content_hash: str
    page: int | None = None
    page_end: int | None = None
    section: str | None = None
    included: bool = True
    include_reason: str = "selected"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FileReceipt:
    """Receipt for a single file."""

    source: Source
    text: str
    chunks: list[Chunk] = field(default_factory=list)
    page_count: int = 0
    section_count: int = 0
    statement_count: int = 0
    included_chars: int = 0
    included_chunk_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_version": RECEIPT_VERSION,
            "source": self.source.to_dict(),
            "page_count": self.page_count,
            "section_count": self.section_count,
            "statement_count": self.statement_count,
            "extracted_char_count": len(self.text),
            "included_char_count": self.included_chars,
            "included_chunk_ids": self.included_chunk_ids,
            "warnings": self.warnings,
            "chunks": [c.to_dict() for c in self.chunks],
        }


@dataclass
class Receipt:
    """Receipt for a whole submission, which may contain several files."""

    receipt_id: str
    created_at: str | None
    pipeline: str
    files: list[FileReceipt] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_version": RECEIPT_VERSION,
            "receipt_id": self.receipt_id,
            "created_at": self.created_at,
            "pipeline": self.pipeline,
            "notes": self.notes,
            "files": [f.to_dict() for f in self.files],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @property
    def total_warnings(self) -> int:
        return sum(len(f.warnings) for f in self.files)


def chunk_text(
    text: str,
    source: Source,
    *,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    include: bool = True,
    include_reason: str = "selected",
) -> list[Chunk]:
    """Split text into chunks aligned to statement boundaries.

    Statement alignment matters: if a chunk boundary falls mid-obligation
    the model receives a truncated requirement, and the receipt would then
    record a chunk that never represented the document faithfully.
    """
    if not text.strip():
        return []

    page_of = _page_lookup(source)
    statements = segment_statements(text, source)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_start = 0
    current_end = 0
    index = 0

    def flush() -> None:
        nonlocal current, current_start, current_end, index
        if not current:
            return
        body = "\n".join(current)
        start = current_start
        end = current_end
        chunks.append(
            Chunk(
                chunk_id=f"{source.source_id}#c{index:03d}",
                source_id=source.source_id,
                text=body,
                start=start,
                end=end,
                char_count=len(body),
                content_hash=sha256_text(body),
                page=page_of(start),
                page_end=page_of(max(start, end - 1)),
                included=include,
                include_reason=include_reason,
            )
        )
        index += 1
        current = []
        current_start = end

    for statement in statements:
        if not current:
            current_start = statement.start
        if current and sum(len(p) for p in current) + len(statement.text) > max_chars:
            flush()
            current_start = statement.start
        current.append(statement.text)
        current_end = statement.end
    flush()

    if not chunks:
        chunks.append(
            Chunk(
                chunk_id=f"{source.source_id}#c000",
                source_id=source.source_id,
                text=text,
                start=0,
                end=len(text),
                char_count=len(text),
                content_hash=sha256_text(text),
                page=page_of(0),
                page_end=page_of(max(0, len(text) - 1)),
                included=include,
                include_reason=include_reason,
            )
        )
    return chunks


def _page_lookup(source: Source):
    """Build a function that maps a character offset to a page number.

    The page boundaries come from the extractor, not from counting
    form feeds, because the extracted text of a PDF is page-joined
    without a form feed. When no page information is available the
    whole document is treated as one page, which is the honest answer
    for a plain text file.
    """
    marks = list(source.page_marks)
    if not marks:
        return lambda offset: 1

    def page_of(offset: int) -> int:
        page = 1
        for index, mark in enumerate(marks, start=1):
            if offset >= mark:
                page = index
            else:
                break
        return page

    return page_of
