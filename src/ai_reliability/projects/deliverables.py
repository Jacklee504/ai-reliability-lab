"""Project 4: AI deliverable contract checker.

Problem
-------
An AI output can look complete while missing something the user asked
for: a required section, a count, a format, an explanation per item, a
named chapter. The output reads fluently, so the gap is easy to miss.

Approach
--------
Two stages, deliberately separated:

1. **Requirement extraction.** The user request is turned into explicit
   requirement records. Each record says what is required and whether it
   can be checked *deterministically* (a count, a heading, a required
   field) or needs *semantic judgement* (is this explanation adequate).

2. **Checking.** Deterministic requirements are checked with structural
   parsing. Semantic requirements are reported as `cannot_verify` until
   a human or a separate semantic checker evaluates them.

The key discipline: the checker never invents requirements. Every
requirement in the output must trace back to something in the request.
Requirements are exposed as an editable list so a user can correct the
extraction before treating it as authoritative.

Known limitations
-----------------
* Requirement extraction from free-form prose is the weakest link. A
  request that says "make it good" yields no requirements, and the
  checker will report nothing, which is honest but not helpful.
* Deterministic checks confirm *structure*, not quality. A deliverable
  with 50 questions of nonsense passes the count check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.claims import extract_qualifiers
from ..core.types import (
    CaseResult,
    CheckStatus,
    EvalCase,
    Finding,
    Method,
)

#: Words that mark a requirement as structural.
_STRUCTURAL_HINTS = (
    "section", "chapter", "heading", "part", "appendix", "table", "list",
    "format", "structure", "field", "column", "row", "header", "title",
)

#: Verbs that introduce a requirement.
_REQUIREMENT_VERBS = (
    "include", "provide", "write", "create", "produce", "give", "add",
    "contain", "have", "must", "should", "ensure", "make sure", "list",
    "cover", "explain", "answer", "label", "mark", "name", "use",
)

#: Number words, so that "six chapters" is as checkable as "6 chapters".
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "hundred": 100,
}

#: Quantities that can be checked exactly. The first element of each pair
#: is the unit pattern, the second is the canonical unit name used when
#: dispatching to a checker.
_COUNT_UNITS = [
    (r"questions?", "question"),
    (r"chapters?|sections?|parts?|units?|lessons?|modules?", "chapter"),
    (r"options?", "option"),
    (r"answers?|explanations?|items?|steps?", "item"),
    (r"words?", "word"),
    (r"paragraphs?", "paragraph"),
]

_NUMBER_TOKEN = (
    r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"fifteen|twenty|thirty|forty|fifty|hundred)"
)

_COUNT_PATTERNS = [
    re.compile(rf"\b{_NUMBER_TOKEN}\s+({units})\b", re.IGNORECASE)
    for units, _name in _COUNT_UNITS
]

#: Canonical unit name for each pattern, used for the check dispatch.
_UNIT_NAMES = [name for _units, name in _COUNT_UNITS]

#: Phrases that require an answer key / explanations.
_ANSWER_KEY = re.compile(r"\banswer\s*key\b", re.IGNORECASE)
_EXPLANATION = re.compile(r"\bexplanation\b", re.IGNORECASE)
_PER_ITEM = re.compile(r"\bfor\s+every\b|\bfor\s+each\b|\bper\s+(?:item|question|answer)\b", re.IGNORECASE)

#: Marker for a required section named in the request.
_NAMED_SECTION = re.compile(
    r"\b(?:section|chapter|part|heading)\s+(?:titled\s+|named\s+|called\s+)?[\"']?([A-Za-z0-9][\w \-/&()']{0,60}?)[\"']?(?=\s*(?:,|;|\.|\band\b|$))",
    re.IGNORECASE,
)

#: Marker for "in markdown", "as json", "as a table" and similar.
_FORMAT_PHRASE = re.compile(
    r"\b(?:in|as|using)\s+(markdown|json|yaml|csv|html|a table|plain text|a bulleted list|a list)\b",
    re.IGNORECASE,
)


@dataclass
class Requirement:
    """One requirement extracted from the user's request."""

    requirement_id: str
    text: str
    kind: str  # "count" | "section" | "format" | "content" | "coverage"
    check: str  # "deterministic" | "semantic"
    target: str = ""
    count: int | None = None
    unit: str = ""
    source_quote: str = ""
    note: str = ""
    comparator: str = "exactly"

    @classmethod
    def from_dict(cls, entry: dict[str, Any]) -> "Requirement":
        """Build a requirement from an edited JSON record.

        A user editing the requirement list may add a field, or leave
        one out. Unknown keys are ignored and missing keys take their
        default, so a small edit does not break the recheck.
        """
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in entry.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "text": self.text,
            "kind": self.kind,
            "check": self.check,
            "target": self.target,
            "count": self.count,
            "unit": self.unit,
            "source_quote": self.source_quote,
            "note": self.note,
            "comparator": self.comparator,
        }


@dataclass
class RequirementCheck:
    """The result of checking one requirement."""

    requirement: Requirement
    status: CheckStatus
    observed: str = ""
    expected: str = ""
    detail: str = ""
    method: Method = Method.DETERMINISTIC

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement": self.requirement.to_dict(),
            "status": self.status.value,
            "observed": self.observed,
            "expected": self.expected,
            "detail": self.detail,
            "method": self.method.value,
        }


@dataclass
class DeliverableReport:
    """The full result of checking a deliverable against its request."""

    request: str
    deliverable: str
    requirements: list[Requirement] = field(default_factory=list)
    checks: list[RequirementCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> list[RequirementCheck]:
        return [c for c in self.checks if c.status is CheckStatus.PASSED]

    @property
    def failed(self) -> list[RequirementCheck]:
        return [c for c in self.checks if c.status is CheckStatus.FAILED]

    @property
    def cannot_verify(self) -> list[RequirementCheck]:
        return [c for c in self.checks if c.status is CheckStatus.CANNOT_VERIFY]
    @property
    def verdict(self) -> str:
        # No requirements means nothing was checked. That is not a pass:
        # reporting "passed" for a vague request would be a false
        # assurance, which is the failure mode this project exists to
        # prevent.
        if not self.requirements:
            return "cannot_verify"
        if self.failed:
            return "failed"
        if self.cannot_verify:
            return "cannot_verify"
        return "passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "verdict": self.verdict,
            "requirement_count": len(self.requirements),
            "passed": len(self.passed),
            "failed": len(self.failed),
            "cannot_verify": len(self.cannot_verify),
            "requirements": [r.to_dict() for r in self.requirements],
            "checks": [c.to_dict() for c in self.checks],
            "notes": self.notes,
        }

    def render(self) -> str:
        lines = [
            f"Deliverable contract check: {self.verdict}",
            f"  requirements: {len(self.requirements)}"
            f"  passed={len(self.passed)} failed={len(self.failed)}"
            f" cannot_verify={len(self.cannot_verify)}",
        ]
        for check in self.checks:
            marker = {
                CheckStatus.PASSED: "PASS",
                CheckStatus.FAILED: "FAIL",
                CheckStatus.CANNOT_VERIFY: "SKIP",
            }[check.status]
            # The requirement ID is shown so a reviewer can tell which
            # line of the edited list this result belongs to, even after
            # the text or the count has been changed.
            lines.append(
                f"  [{marker}] {check.requirement.requirement_id}:"
                f" {check.requirement.text}"
            )
            if check.status is not CheckStatus.PASSED:
                lines.append(f"         expected {check.expected}, observed {check.observed}")
                if check.detail:
                    lines.append(f"         {check.detail}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Requirement extraction
# ---------------------------------------------------------------------------


def _parse_number(token: str) -> int | None:
    """Parse a count token, accepting digits and number words.

    Returns ``None`` when the token is not a number, so a request like
    "50 questions" and "fifty questions" are treated the same way and a
    request with no count produces no count requirement at all.
    """
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)
