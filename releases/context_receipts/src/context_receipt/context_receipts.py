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

from .core.fixtures import project_fixture_dir
from .core.sources import load_source
from .core.spans import segment_statements, split_sections
from .core.types import (
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

def build_file_receipt(
    path: str | Path,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    include_file: bool = True,
) -> FileReceipt:
    """Build the receipt for one file.

    A file that failed to extract still produces a receipt, and that
    receipt is the point: the user can see that the pipeline submitted
    nothing for this file instead of assuming it was summarised.
    """
    source, text = load_source(path)
    warnings = list(source.warnings)

    # The page count comes from the extractor. For a PDF that is the
    # number of pages the reader saw; for anything else the document is
    # one page of text, or zero pages when nothing was extracted.
    if source.page_count:
        page_count = source.page_count
    else:
        page_count = 1 if text.strip() else 0

    sections = split_sections(text) if text.strip() else []
    statements = segment_statements(text, source) if text.strip() else []

    chunks = chunk_text(
        text,
        source,
        max_chars=chunk_chars,
        include=include_file,
        include_reason="selected" if include_file else "excluded_by_caller",
    )

    included = [c for c in chunks if c.included]
    included_chars = sum(c.char_count for c in included)

    if source.extraction_status is ExtractionStatus.EMPTY:
        warnings.append("no text was extracted from this file")
    if not included:
        warnings.append("no chunks were selected for submission")

    return FileReceipt(
        source=source,
        text=text,
        chunks=chunks,
        page_count=page_count,
        section_count=len(sections),
        statement_count=len(statements),
        included_chars=included_chars,
        included_chunk_ids=[c.chunk_id for c in included],
        warnings=warnings,
    )


def build_receipt(
    paths: Iterable[str | Path],
    *,
    pipeline: str = "ai-reliability-lab/context-receipt",
    created_at: str | None = None,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
) -> Receipt:
    """Build a receipt covering every listed file."""
    files = [build_file_receipt(p, chunk_chars=chunk_chars) for p in paths]
    receipt_id = sha256_text(
        "|".join(f.source.content_hash or f.source.source_id for f in files)
    )[:12]
    return Receipt(
        receipt_id=receipt_id,
        created_at=created_at,
        pipeline=pipeline,
        files=files,
    )


def file_key(file_receipt: FileReceipt) -> str:
    """Stable key used to match the same logical file across receipts.

    The file *name* is used, not the content hash, because the question the
    diff answers is "did this file change", which requires recognising that
    two different files are versions of the same thing. Matching on the
    name also lets a receipt be compared across directories, which is how
    a stale-version check is normally run.
    """
    source = file_receipt.source
    reference = source.path or source.identifier or source.source_id
    return Path(reference).name or reference




def _file_diff(before: FileReceipt, after: FileReceipt) -> list[str]:
    """Human-readable differences between two file receipts."""
    lines: list[str] = []
    a = before.source
    b = after.source

    if a.content_hash != b.content_hash:
        lines.append(f"content hash changed: {a.content_hash[:12]} -> {b.content_hash[:12]}")
    else:
        lines.append("content hash unchanged")
    if a.version != b.version:
        lines.append(f"version changed: {a.version} -> {b.version}")
    if a.extraction_status != b.extraction_status:
        lines.append(
            f"extraction status changed: {a.extraction_status.value} -> {b.extraction_status.value}"
        )
    if len(before.text) != len(after.text):
        lines.append(f"extracted characters: {len(before.text)} -> {len(after.text)}")
    if before.page_count != after.page_count:
        lines.append(f"page count: {before.page_count} -> {after.page_count}")
    if before.section_count != after.section_count:
        lines.append(f"section count: {before.section_count} -> {after.section_count}")
    if before.included_chars != after.included_chars:
        lines.append(
            f"submitted characters: {before.included_chars} -> {after.included_chars}"
        )
    if before.included_chunk_ids != after.included_chunk_ids:
        removed = set(before.included_chunk_ids) - set(after.included_chunk_ids)
        added = set(after.included_chunk_ids) - set(before.included_chunk_ids)
        if removed:
            lines.append(f"chunks no longer submitted: {sorted(removed)}")
        if added:
            lines.append(f"chunks newly submitted: {sorted(added)}")
    for warning in after.warnings:
        if warning not in before.warnings:
            lines.append(f"new warning: {warning}")
    return lines


def diff_receipts(before: Receipt, after: Receipt) -> dict[str, Any]:
    """Compare two receipts and describe what a user would notice.

    Files are matched by path, so a file that was replaced by a newer
    version reads as *changed* rather than as one removal plus one
    addition. That distinction is the whole point of the diff.
    """
    before_by_key = {file_key(f): f for f in before.files}
    after_by_key = {file_key(f): f for f in after.files}

    changes: dict[str, Any] = {}
    for key in sorted(set(before_by_key) | set(after_by_key)):
        old = before_by_key.get(key)
        new = after_by_key.get(key)
        if old is None and new is not None:
            changes[key] = {
                "status": "added",
                "lines": [f"file added: {new.source.path}"],
            }
        elif new is None and old is not None:
            changes[key] = {
                "status": "removed",
                "lines": [f"file removed: {old.source.path}"],
            }
        else:
            assert old is not None and new is not None
            lines = _file_diff(old, new)
            unchanged = bool(lines) and lines[0] == "content hash unchanged"
            changes[key] = {
                "status": "unchanged" if unchanged else "changed",
                "lines": lines,
            }

    return {
        "before_receipt_id": before.receipt_id,
        "after_receipt_id": after.receipt_id,
        "files": changes,
        "identical": all(c["status"] == "unchanged" for c in changes.values()),
    }


def render_receipt(receipt: Receipt) -> str:
    """Human-readable rendering of a receipt."""
    lines = [
        f"Receipt {receipt.receipt_id}  pipeline={receipt.pipeline}",
        f"Files: {len(receipt.files)}   warnings: {receipt.total_warnings}",
    ]
    for file_receipt in receipt.files:
        source = file_receipt.source
        lines.append("")
        lines.append(f"  {source.path or source.identifier}")
        lines.append(f"    hash        {source.content_hash[:16] or '(none)'}")
        lines.append(f"    media type  {source.media_type.value}")
        lines.append(
            f"    extraction  {source.extraction_status.value}"
            f"  pages={file_receipt.page_count} sections={file_receipt.section_count}"
        )
        lines.append(
            f"    characters  extracted={len(file_receipt.text)} submitted={file_receipt.included_chars}"
        )
        lines.append(
            f"    chunks      {len(file_receipt.chunks)}"
            f" ({len(file_receipt.included_chunk_ids)} submitted)"
        )
        for warning in file_receipt.warnings:
            lines.append(f"    WARNING     {warning}")
    return "\n".join(lines)


def render_diff(diff: dict[str, Any]) -> str:
    """Human-readable rendering of a receipt diff."""
    lines = [
        f"Receipt diff: {diff['before_receipt_id']} -> {diff['after_receipt_id']}",
    ]
    if diff["identical"]:
        lines.append("  No differences: the same text was submitted both times.")
        return "\n".join(lines)
    for key, change in diff["files"].items():
        lines.append(f"  {key}  [{change['status']}]")
        for line in change["lines"]:
            lines.append(f"    {line}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


@dataclass
class ReceiptRunner:
    """Runs a receipt evaluation case.

    Receipt cases are checked on *receipt properties* rather than on
    verdicts: extraction status, how much text was submitted, which
    warnings appeared, and, for diff cases, which kinds of change the
    diff reported.
    """

    def evaluate_case(self, case: EvalCase) -> CaseResult:
        payload = case.input
        expected = case.expected
        files = payload.get("files", [])
        resolved = [self._resolve(f) for f in files]

        details: dict[str, Any] = {}
        problems: list[str] = []

        if payload.get("diff_against"):
            before = build_receipt([self._resolve(f) for f in payload["diff_against"]])
            after = build_receipt(resolved)
            diff = diff_receipts(before, after)
            details["diff"] = diff
            self._check_diff(expected, diff, problems)
        else:
            for path in resolved:
                receipt = build_file_receipt(path)
                details[Path(path).name] = receipt.to_dict()
                self._check_file(expected, receipt, problems)

        return CaseResult(
            case_id=case.case_id,
            project=case.project,
            predicted={
                "verdict": "passed" if not problems else "failed",
                "problems": problems,
                "details": details,
            },
            expected=expected,
            label_origin=case.label_origin,
            split=case.split,
            correct=not problems,
            notes="; ".join(problems) if problems else None,
        )

    @staticmethod
    def _resolve(path: str) -> str:
        candidate = Path(path)
        if candidate.exists():
            return str(candidate)
        in_fixtures = project_fixture_dir("context_receipts") / path
        if in_fixtures.exists():
            return str(in_fixtures)
        return str(path)

    @staticmethod
    def _check_file(expected: dict[str, Any], receipt: FileReceipt, problems: list[str]) -> None:
        source = receipt.source
        if "extraction_status" in expected:
            if source.extraction_status.value != expected["extraction_status"]:
                problems.append(
                    f"extraction_status was {source.extraction_status.value},"
                    f" expected {expected['extraction_status']}"
                )
        if "submitted_char_count_gt" in expected:
            if receipt.included_chars <= expected["submitted_char_count_gt"]:
                problems.append(
                    f"submitted_char_count {receipt.included_chars}"
                    f" is not greater than {expected['submitted_char_count_gt']}"
                )
        if "submitted_char_count_eq" in expected:
            if receipt.included_chars != expected["submitted_char_count_eq"]:
                problems.append(
                    f"submitted_char_count {receipt.included_chars}"
                    f" != {expected['submitted_char_count_eq']}"
                )
        if "warning_containing" in expected:
            blob = " ".join(receipt.warnings).lower()
            if expected["warning_containing"].lower() not in blob:
                problems.append(
                    f"no warning containing {expected['warning_containing']!r};"
                    f" warnings were {receipt.warnings}"
                )
        if expected.get("no_warnings") and receipt.warnings:
            problems.append(f"expected no warnings, got {receipt.warnings}")
        if "chunk_count_gte" in expected:
            if len(receipt.included_chunk_ids) < expected["chunk_count_gte"]:
                problems.append(
                    f"chunk count {len(receipt.included_chunk_ids)}"
                    f" < {expected['chunk_count_gte']}"
                )
        if "section_count_gte" in expected:
            if receipt.section_count < expected["section_count_gte"]:
                problems.append(
                    f"section count {receipt.section_count} < {expected['section_count_gte']}"
                )
        if "content_hash_prefix" in expected:
            if not source.content_hash.startswith(expected["content_hash_prefix"]):
                problems.append(
                    f"content hash {source.content_hash[:16]}"
                    f" does not start with {expected['content_hash_prefix']}"
                )

    @staticmethod
    def _check_diff(expected: dict[str, Any], diff: dict[str, Any], problems: list[str]) -> None:
        blob = str(diff).lower()
        for needle in expected.get("diff_must_include", []):
            if needle.lower() not in blob:
                problems.append(f"diff did not mention {needle!r}")
        if expected.get("identical") is True and not diff["identical"]:
            problems.append("expected identical receipts, but they differ")
        if expected.get("identical") is False and diff["identical"]:
            problems.append("expected differing receipts, but they are identical")


def build_cases() -> list[EvalCase]:
    """Load the receipt project's evaluation cases."""
    from .core.fixtures import load_project_cases

    return load_project_cases("context_receipts")


def build_runner(provider: Any = None) -> ReceiptRunner:
    """Factory used by the project registry."""
    return ReceiptRunner()
