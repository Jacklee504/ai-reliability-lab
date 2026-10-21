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


def load_source(path: str | os.PathLike[str]) -> tuple[Source, str]:
    """Load a file into a :class:`Source` record and its extracted text.

    The returned text is what the pipeline would actually submit. For PDFs
    it is page-joined extracted text; for everything else it is the decoded
    file body.
    """
    p = Path(path)
    media_type = _media_type(p)

    try:
        data = p.read_bytes()
    except FileNotFoundError:
        source = Source(
            source_id=short_id("src", str(p)),
            path=str(p),
            content_hash="",
            media_type=media_type,
            extraction_status=ExtractionStatus.FAILED,
            warnings=("file not found",),
        )
        return source, ""
    except OSError as exc:
        source = Source(
            source_id=short_id("src", str(p)),
            path=str(p),
            content_hash="",
            media_type=media_type,
            extraction_status=ExtractionStatus.FAILED,
            warnings=(f"unreadable: {type(exc).__name__}: {exc}",),
        )
        return source, ""

    digest = sha256_bytes(data)

    page_count = 0
    page_marks: tuple[int, ...] = ()
    if media_type is MediaType.PDF:
        text, warnings, status, page_count, page_marks = _extract_pdf(data)
    elif media_type is MediaType.UNKNOWN:
        # Unknown suffix: try UTF-8 decode, and report clearly if it fails.
        try:
            text = data.decode("utf-8")
            warnings: tuple[str, ...] = ("unknown media type; file decoded as utf-8 text",)
            status = ExtractionStatus.OK
        except UnicodeDecodeError as exc:
            text = ""
            warnings = (f"not decodable as utf-8: {exc}",)
            status = ExtractionStatus.UNSUPPORTED
    else:
        try:
            text = data.decode("utf-8")
            warnings = ()
            status = ExtractionStatus.OK
        except UnicodeDecodeError as exc:
            text = data.decode("utf-8", errors="replace")
            warnings = (f"utf-8 decode replaced invalid bytes: {exc}",)
            status = ExtractionStatus.OK

    if status is ExtractionStatus.OK and not text.strip():
        status = ExtractionStatus.EMPTY
        warnings = tuple(warnings) + ("file is empty",)

    source = Source(
        source_id=short_id("src", str(p)),
        path=str(p),
        content_hash=digest,
        media_type=media_type,
        extraction_status=status,
        warnings=warnings,
        char_count=len(text),
        page_count=page_count,
        page_marks=page_marks,
    )
    return source, text


def source_from_text(
    text: str,
    identifier: str,
    media_type: MediaType = MediaType.TEXT,
    version: str | None = None,
) -> tuple[Source, str]:
    """Build an in-memory source, used by fixtures that ship as literals."""
    source = Source(
        source_id=short_id("src", identifier, sha256_text(text)),
        identifier=identifier,
        content_hash=sha256_text(text),
        version=version,
        media_type=media_type,
        extraction_status=ExtractionStatus.OK,
        char_count=len(text),
    )
    return source, text


def build_span(source: Source, text: str, start: int, end: int, **kwargs) -> Span:
    """Create a span whose ``text`` is exactly the slice it points at."""
    clipped_end = min(end, len(text))
    clipped_start = max(0, min(start, clipped_end))
    return Span(
        span_id=short_id("sp", source.source_id, str(clipped_start), str(clipped_end)),
        source_id=source.source_id,
        text=text[clipped_start:clipped_end],
        start=clipped_start,
        end=clipped_end,
        **kwargs,
    )
