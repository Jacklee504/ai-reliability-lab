"""Source segmentation into candidate evidence passages.

Segmentation is deterministic, so the same source always produces the
same candidate passages. That is what makes the "safe to delete"
synthetic transformations reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import Source, Span, short_id

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


@dataclass(frozen=True)
class Statement:
    """A candidate source statement, with the exact span it came from."""

    statement_id: str
    text: str
    start: int
    end: int
    section: str | None = None
    kind: str = "statement"

    def to_span(self, source: Source) -> Span:
        return Span(
            span_id=short_id("sp", source.source_id, str(self.start), str(self.end)),
            source_id=source.source_id,
            text=self.text,
            start=self.start,
            end=self.end,
            section=self.section,
            kind=self.kind,
        )


def _line_offsets(text: str) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        out.append((pos, pos + len(line), line))
        pos += len(line)
    return out


def _is_title_case(text: str) -> bool:
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


def split_sections(text: str) -> list[Section]:
    """Split text into sections based on headings."""
    sections: list[Section] = []
    current_title = ""
    current_start = 0
    for start, end, line in _line_offsets(text):
        if is_heading(line):
            if current_start < start:
                sections.append(Section(current_title, current_start, start))
            current_title = line.strip().rstrip(":").strip()
            current_start = end
    if current_start < len(text) or not sections:
        sections.append(Section(current_title, current_start, len(text)))
    return [s for s in sections if s.end > s.start] or [Section("", 0, len(text))]


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """Split a block of text into (start, end, text) fragments."""
    if not text.strip():
        return []
    out: list[tuple[int, int, str]] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.start()
        fragment = text[cursor:end]
        stripped = fragment.strip()
        if stripped:
            lead = fragment.index(stripped[0])
            out.append((cursor + lead, cursor + lead + len(stripped), stripped))
        cursor = match.end()
    tail = text[cursor:]
    stripped = tail.strip()
    if stripped:
        lead = tail.index(stripped[0])
        out.append((cursor + lead, cursor + lead + len(stripped), stripped))
    return out


def segment_statements(text: str, source: Source) -> list[Statement]:
    """Split a source into candidate statements."""
    if not text.strip():
        return []
    sections = split_sections(text)
    statements: list[Statement] = []
    for section in sections:
        body = text[section.start : section.end]
        base = section.start
        for line_start, _line_end, line in _line_offsets(body):
            raw = line.strip()
            if not raw:
                continue
            bullet = _BULLET.match(line)
            if bullet and not is_heading(line):
                content = line[bullet.end():].strip()
                if not content:
                    continue
                offset_in_line = line.index(content, bullet.end())
                start = base + line_start + offset_in_line
                end = start + len(content)
                statements.append(
                    Statement(
                        statement_id=short_id("st", source.source_id, str(start), str(end)),
                        text=text[start:end],
                        start=start,
                        end=end,
                        section=section.title or None,
                        kind="list_item",
                    )
                )
                continue
            line_body = line.strip()
            lead = len(line) - len(line.lstrip())
            para_start = base + line_start + lead
            for rel_start, _rel_end, fragment in sentence_spans(line_body):
                start = para_start + rel_start
                end = start + len(fragment)
                statements.append(
                    Statement(
                        statement_id=short_id("st", source.source_id, str(start), str(end)),
                        text=text[start:end],
                        start=start,
                        end=end,
                        section=section.title or None,
                        kind="statement",
                    )
                )
    seen: set[tuple[int, int]] = set()
    unique: list[Statement] = []
    for st in statements:
        key = (st.start, st.end)
        if key in seen:
            continue
        seen.add(key)
        unique.append(st)
    return unique
