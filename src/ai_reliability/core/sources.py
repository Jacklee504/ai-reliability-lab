"""Source ingestion: turn a file into a :class:`Source` plus its text.

The lab uses only synthetic or openly licensed fixtures, so ingestion is
deliberately simple. The important behaviour is *honest failure*: an
unreadable, empty, or unsupported file produces an ``ExtractionStatus``
that says so instead of silently yielding empty text, because a silent
empty extraction is indistinguishable from a document with no content
and is exactly the failure mode the context-receipt project exists to
expose.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

from .types import (
    ExtractionStatus,
    MediaType,
    Source,
    Span,
    sha256_bytes,
    sha256_text,
    short_id,
)

TEXT_SUFFIXES = {".txt", ".text", ".log", ".rst"}
MARKDOWN_SUFFIXES = {".md", ".markdown"}
CODE_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".h", ".sh"}


def _media_type(path: Path) -> MediaType:
    suffix = path.suffix.lower()
    if suffix in MARKDOWN_SUFFIXES:
        return MediaType.MARKDOWN
    if suffix == ".csv":
        return MediaType.CSV
    if suffix == ".json":
        return MediaType.JSON
    if suffix == ".pdf":
        return MediaType.PDF
    if suffix in CODE_SUFFIXES:
        return MediaType.CODE
    if suffix in TEXT_SUFFIXES:
        return MediaType.TEXT
    return MediaType.UNKNOWN


def _extract_pdf(
    data: bytes,
) -> tuple[str, tuple[str, ...], ExtractionStatus, int, tuple[int, ...]]:
    """Extract PDF text with pypdf when available.

    Returns extracted text, warnings, status, page count, and the
    character offset where each page begins. A PDF with no extractable
    text is reported as ``empty`` with a warning rather than as a
    successful extraction of nothing.
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        return (
            "",
            (f"pdf extraction unavailable: {type(exc).__name__}: {exc}",),
            ExtractionStatus.UNSUPPORTED,
            0,
            (),
        )

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        return (
            "",
            (f"pdf parse failed: {type(exc).__name__}: {exc}",),
            ExtractionStatus.FAILED,
            0,
            (),
        )

    pages: list[str] = []
    warnings: list[str] = []
    page_marks: list[int] = []
    cursor = 0
    for index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            page_text = ""
            warnings.append(f"page {index} extraction failed: {type(exc).__name__}")
        if index > 1:
            cursor += 2  # the "\n\n" join between pages
        page_marks.append(cursor)
        cursor += len(page_text)
        pages.append(page_text)

    full = "\n\n".join(pages)
    if not full.strip():
        return (
            full,
            tuple(warnings + ["no extractable text (image-only or empty PDF)"]),
            ExtractionStatus.EMPTY,
            len(pages),
            tuple(page_marks),
        )

    status = ExtractionStatus.PARTIAL if warnings else ExtractionStatus.OK
    return full, tuple(warnings), status, len(pages), tuple(page_marks)
