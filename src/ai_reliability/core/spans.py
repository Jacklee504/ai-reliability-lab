"""Text segmentation: sentences, sections, and page-aware spans.

Statement segmentation is the foundation for the claim auditor, the
omission auditor, and the citation checker. It is intentionally
deterministic so that the same source always produces the same candidate
units, which is what makes the omission "safe to delete" synthetic
transformations reproducible.

Two kinds of segmentation live here:

``segment_statements``
    Splits a source into candidate factual statements. This is what the
    omission auditor ranks and what the claim auditor aligns against.

``split_sections``
    Splits a document into headed sections, so a source statement can be
    reported with the section it came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import Source, Span, short_id

# Sentence boundary: a terminator followed by whitespace and a capital or
# digit. Keeps "Dr. Smith" and decimal numbers intact.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9“\"(])")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_HEADING_MARKDOWN = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_HEADING_COLON = re.compile(r"^\s{0,3}[A-Z][A-Za-z0-9 \-/&()']{2,80}:\s*$")
_HEADING_ALLCAPS = re.compile(r"^\s{0,3}[A-Z0-9][A-Z0-9 \-/&()']{3,80}\s*$")
_HEADING_NUMBERED = re.compile(r"^\s{0,3}(\d+)[.)]\s+(\S.*)$")
_SECTION_WORD = re.compile(
    r"^\s*(?:section|chapter|part|appendix|schedule|exhibit|article)\s+"
    r"[0-9IVXLC]+[.:)]?\s",
    re.IGNORECASE,
)

#: Words that stay lowercase inside a title-cased heading.
_MINOR_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into",
    "nor", "of", "on", "or", "per", "the", "to", "under", "via", "with",
}


@dataclass(frozen=True)
class Section:
    """A headed region of a document."""

    title: str
    start: int
    end: int
    page: int | None = None


@dataclass(frozen=True)
class Statement:
    """A candidate source statement, with the exact span it came from."""

    statement_id: str
    text: str
    start: int
    end: int
    section: str | None = None
    page: int | None = None
    kind: str = "statement"

    def to_span(self, source: Source) -> Span:
        return Span(
            span_id=short_id("sp", source.source_id, str(self.start), str(self.end)),
            source_id=source.source_id,
            text=self.text,
            start=self.start,
            end=self.end,
            section=self.section,
            page=self.page,
            kind=self.kind,
        )


def _line_offsets(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, line) for each line, including its terminator."""
    out: list[tuple[int, int, str]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        out.append((pos, pos + len(line), line))
        pos += len(line)
    return out


def _is_title_case(text: str) -> bool:
    """Heuristic: does this look like a heading rather than a sentence?

    A heading starts with a capital and has mostly capitalised words,
    ignoring articles and prepositions. This is what separates
    "1. Fees and Payment" (a heading) from "1. the fee is due" (a list
    item that happens to start with a number).
    """
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        return False
    if not words[0][:1].isupper():
        return False
    content = [w for w in words if w.lower().strip(".,:;") not in _MINOR_WORDS]
    if not content:
        return False
    capitalised = sum(1 for w in content if w[:1].isupper())
    return capitalised / len(content) >= 0.6


def is_heading(line: str) -> bool:
    """True when a line looks like a section heading."""
    stripped = line.strip()
    if not stripped:
        return False
    if _HEADING_MARKDOWN.match(line):
        return True
    if _SECTION_WORD.match(line):
        return True
    if _HEADING_COLON.match(line):
        return True
    if _HEADING_ALLCAPS.match(line):
        return True
    numbered = _HEADING_NUMBERED.match(line)
    if numbered:
        rest = numbered.group(2).strip()
        if len(stripped) <= 80 and _is_title_case(rest):
            return True
    return False
