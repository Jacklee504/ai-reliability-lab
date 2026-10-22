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

from .core.claims import extract_qualifiers
from .core.types import (
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


def _sentences(text: str) -> list[str]:
    from .core.spans import sentence_spans

    return [fragment for _s, _e, fragment in sentence_spans(text)]


def extract_requirements(request: str) -> list[Requirement]:
    """Turn a user request into explicit requirement records.

    Every requirement keeps the exact quote it came from, so a user can
    check that nothing was invented. When no requirement can be found the
    result is empty, and the caller reports that as "cannot verify" rather
    than passing the deliverable.
    """
    requirements: list[Requirement] = []
    counter = 0

    def add(**kwargs: Any) -> None:
        nonlocal counter
        counter += 1
        kwargs["requirement_id"] = f"req_{counter:02d}"
        requirements.append(Requirement(**kwargs))

    for sentence in _sentences(request):
        lowered = sentence.lower()
        if not any(verb in lowered for verb in _REQUIREMENT_VERBS):
            continue

        matched_count = False
        for pattern, canonical_unit in zip(_COUNT_PATTERNS, _UNIT_NAMES):
            for match in pattern.finditer(sentence):
                number = _parse_number(match.group(1))
                if number is None:
                    continue
                prefix = sentence[max(0, match.start() - 24):match.start()].lower()
                if re.search(r"(?:at least|no fewer than|minimum of)\s*$", prefix):
                    comparator = "at_least"
                elif re.search(r"(?:at most|no more than|up to)\s*$", prefix):
                    comparator = "at_most"
                else:
                    comparator = "exactly"
                add(
                    text=f"must contain {number} {canonical_unit}",
                    kind="count",
                    check="deterministic",
                    count=number,
                    unit=canonical_unit,
                    comparator=comparator,
                    source_quote=sentence,
                )
                matched_count = True

        for match in _NAMED_SECTION.finditer(sentence):
            name = match.group(1).strip(" .,")
            if not name or len(name) < 2:
                continue
            add(
                text=f"must contain a section named {name!r}",
                kind="section",
                check="deterministic",
                target=name,
                source_quote=sentence,
            )

        format_match = _FORMAT_PHRASE.search(sentence)
        if format_match:
            fmt = format_match.group(1).lower()
            add(
                text=f"must be delivered in {fmt}",
                kind="format",
                check="deterministic",
                target=fmt,
                source_quote=sentence,
            )

        if _ANSWER_KEY.search(sentence):
            add(
                text="must include an answer key",
                kind="content",
                check="deterministic",
                target="answer_key",
                source_quote=sentence,
            )
            if _PER_ITEM.search(sentence):
                add(
                    text="must include an explanation for every answer",
                    kind="content",
                    check="semantic",
                    target="explanation_per_answer",
                    source_quote=sentence,
                    note="an explanation per item needs judgement, not counting",
                )

        if _EXPLANATION.search(sentence) and not _ANSWER_KEY.search(sentence):
            add(
                text="must include explanations",
                kind="content",
                check="semantic",
                target="explanations",
                source_quote=sentence,
            )

        if not matched_count and not format_match and not _ANSWER_KEY.search(sentence):
            # A request verb with no countable or named target: this is a
            # coverage requirement, which needs semantic judgement.
            add(
                text=sentence.strip(),
                kind="coverage",
                check="semantic",
                source_quote=sentence,
            )

    return requirements


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------


#: Headings that introduce a section which is not part of the question
#: list. An answer key repeats the question numbers, so counting it
#: would inflate the question count and hide a missing question.
_NON_QUESTION_SECTIONS = re.compile(
    r"^\s*(?:answer\s*key|answers?|key|solutions?|marking\s*scheme|"
    r"explanations?|rationale|notes?|scoring)\b",
    re.IGNORECASE,
)


def _count_questions(deliverable: str) -> int:
    """Count questions in a deliverable.

    Recognises the common shapes: numbered items, ``Q1:``/``Q. 1``/``Q1)``
    prefixes, and lines ending in a question mark.

    Sections that list answers rather than questions are skipped. An
    answer key repeats the question numbers, so counting it as well
    would make a deliverable with missing questions look complete.
    """
    count = 0
    seen: set[str] = set()
    patterns = [
        re.compile(r"^\s*(?:Q(?:uestion)?\.?\s*)?(\d+)\s*[.):]\s+\S", re.IGNORECASE),
        re.compile(r"^\s*Q(?:uestion)?\.?\s*(\d+)\s*[.):]?\s+\S", re.IGNORECASE),
    ]
    in_non_question_section = False
    for line in deliverable.splitlines():
        if _NON_QUESTION_SECTIONS.match(line):
            in_non_question_section = True
            continue
        if in_non_question_section:
            # A new heading ends the answer-key section.
            if line.strip() and re.match(r"^\s*[A-Z][A-Za-z0-9 \-/&()']{2,80}\s*$", line):
                in_non_question_section = False
            else:
                continue
        for pattern in patterns:
            match = pattern.match(line)
            if match:
                number = match.group(1)
                if number not in seen:
                    seen.add(number)
                    count += 1
                break
        else:
            if line.strip().endswith("?") and len(line.strip()) > 3:
                count += 1
    return count


def _count_options(deliverable: str) -> int:
    """Return the fewest distinct options attached to any question."""
    option_re = re.compile(r"^\s*\(?([A-Za-z])\)?[.):]\s+\S")
    question_re = re.compile(
        r"^\s*(?:Q(?:uestion)?\.?\s*\d+\s*[.):]?\s+\S|\d+\s*[.):]\s+.*\?)",
        re.IGNORECASE,
    )
    counts: list[int] = []
    current: set[str] = set()
    active = False
    for line in deliverable.splitlines():
        if question_re.match(line):
            if active:
                counts.append(len(current))
            current = set()
            active = True
        elif active and (match := option_re.match(line)):
            current.add(match.group(1).upper())
    if active:
        counts.append(len(current))
    return min(counts, default=0)


def _chapters_present(deliverable: str, count: int) -> tuple[int, list[str]]:
    """Count distinct numbered chapters or sections in the deliverable."""
    found: set[str] = set()
    pattern = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:chapter|section|part|unit|module)\s+([0-9]+|[IVXLC]+)\b",
        re.IGNORECASE,
    )
    markdown = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
    for line in deliverable.splitlines():
        match = pattern.match(line)
        if match:
            found.add(match.group(1).upper())
            continue
        heading = markdown.match(line)
        if heading:
            found.add(heading.group(1).strip().upper())
    return len(found), sorted(found)


def _named_section_present(deliverable: str, name: str) -> bool:
    """True when the deliverable contains a heading matching the name."""
    target = re.sub(r"\s+", " ", name.strip().lower())
    for line in deliverable.splitlines():
        stripped = line.strip()
        stripped = re.sub(r"^#{1,6}\s*", "", stripped)
        stripped = stripped.rstrip(":").strip()
        if stripped.lower() == target:
            return True
        if target and target in stripped.lower():
            return True
    return False


def _format_present(deliverable: str, fmt: str) -> bool:
    """Check the requested output format."""
    fmt = fmt.lower()
    if "markdown" in fmt:
        return bool(re.search(r"^\s*#{1,6}\s+\S", deliverable, re.MULTILINE)) or "```" in deliverable
    if "json" in fmt:
        return deliverable.strip().startswith(("{", "["))
    if "csv" in fmt:
        lines = [ln for ln in deliverable.splitlines() if ln.strip()]
        if len(lines) < 2:
            return False
        return lines[0].count(",") == lines[1].count(",")
    if "yaml" in fmt:
        return bool(re.search(r"^\s*[\w.-]+:\s*\S", deliverable, re.MULTILINE))
    if "html" in fmt:
        return "<" in deliverable and ">" in deliverable
    if "table" in fmt:
        return "|" in deliverable
    if "list" in fmt:
        return bool(re.search(r"^\s*[-*•]\s+\S", deliverable, re.MULTILINE))
    if "plain text" in fmt:
        return not re.search(r"^\s*#{1,6}\s+\S", deliverable, re.MULTILINE)
    return False


def _answer_key_present(deliverable: str) -> bool:
    return bool(re.search(r"answer\s*key", deliverable, re.IGNORECASE))


def check_requirement(requirement: Requirement, deliverable: str) -> RequirementCheck:
    """Check one requirement against the deliverable, deterministically."""
    def count_passes(observed_count: int) -> bool:
        if requirement.comparator == "at_least":
            return observed_count >= (requirement.count or 0)
        if requirement.comparator == "at_most":
            return observed_count <= (requirement.count or 0)
        return observed_count == (requirement.count or 0)

    if requirement.kind == "count":
        unit = requirement.unit
        if "question" in unit:
            observed = str(_count_questions(deliverable))
            expected = str(requirement.count)
        elif "option" in unit:
            observed = str(_count_options(deliverable))
            expected = str(requirement.count)
        elif unit.startswith("chapter") or unit.startswith("section") or unit.startswith("part"):
            observed_count, names = _chapters_present(deliverable, requirement.count or 0)
            observed = str(observed_count)
            expected = str(requirement.count)
            status = (
                CheckStatus.PASSED
                if count_passes(observed_count)
                else CheckStatus.FAILED
            )
            return RequirementCheck(
                requirement=requirement,
                status=status,
                observed=observed,
                expected=expected,
                detail=f"found headings: {', '.join(names) if names else '(none)'}",
            )
        else:
            observed = "unknown"
            expected = str(requirement.count)
        status = (
            CheckStatus.PASSED
            if observed != "unknown" and count_passes(int(observed))
            else CheckStatus.FAILED
        )
        return RequirementCheck(
            requirement=requirement,
            status=status,
            observed=observed,
            expected=expected,
            detail=f"counted {observed} {unit} in the deliverable",
        )

    if requirement.kind == "section":
        present = _named_section_present(deliverable, requirement.target)
        return RequirementCheck(
            requirement=requirement,
            status=CheckStatus.PASSED if present else CheckStatus.FAILED,
            observed="present" if present else "absent",
            expected="present",
            detail=f"looked for a heading matching {requirement.target!r}",
        )

    if requirement.kind == "format":
        present = _format_present(deliverable, requirement.target)
        return RequirementCheck(
            requirement=requirement,
            status=CheckStatus.PASSED if present else CheckStatus.FAILED,
            observed="present" if present else "absent",
            expected="present",
            detail=f"looked for {requirement.target} formatting",
        )

    if requirement.kind == "content" and requirement.target == "answer_key":
        present = _answer_key_present(deliverable)
        return RequirementCheck(
            requirement=requirement,
            status=CheckStatus.PASSED if present else CheckStatus.FAILED,
            observed="present" if present else "absent",
            expected="present",
            detail="looked for an answer key section",
        )

    # Semantic requirements: counted as a separate, clearly labelled band.
    return RequirementCheck(
        requirement=requirement,
        status=CheckStatus.CANNOT_VERIFY,
        observed="not checked deterministically",
        expected="judged by a model or a human",
        detail="this requirement needs semantic judgement",
        method=Method.SEMANTIC,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_deliverable(
    request: str, deliverable: str, requirements: list[Requirement] | None = None,
) -> DeliverableReport:
    """Check extracted requirements, or a user-edited requirement list."""
    requirements = list(requirements) if requirements is not None else extract_requirements(request)
    ids = [item.requirement_id for item in requirements]
    if len(ids) != len(set(ids)):
        raise ValueError("requirement IDs must be unique")
    checks = [check_requirement(r, deliverable) for r in requirements]

    notes: list[str] = []
    if not requirements:
        notes.append(
            "no explicit requirements could be extracted from the request;"
            " nothing was checked"
        )
    semantic = [c for c in checks if c.requirement.check == "semantic"]
    if semantic:
        notes.append(
            f"{len(semantic)} requirement(s) need semantic judgement and are"
            " reported as cannot_verify without a model"
        )

    return DeliverableReport(
        request=request,
        deliverable=deliverable,
        requirements=requirements,
        checks=checks,
        notes=notes,
    )


def summarize(report: DeliverableReport) -> dict[str, Any]:
    """Compact summary suitable for the evaluation layer."""
    return {
        "verdict": report.verdict,
        "requirement_count": len(report.requirements),
        "passed": len(report.passed),
        "failed": len(report.failed),
        "cannot_verify": len(report.cannot_verify),
        "failures": [
            {
                "text": c.requirement.text,
                "expected": c.expected,
                "observed": c.observed,
            }
            for c in report.failed
        ],
    }


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


@dataclass
class DeliverableRunner:
    """Runs a deliverable contract evaluation case.

    The runner calls the checker, summarises the outcome, and scores the
    case against its expected label. A deliverable case carries several
    conditions rather than a single label, so scoring is done here
    instead of in the shared layer.

    A deliverable whose only failing requirements are semantic ones is
    reported as ``cannot_verify``, never as a pass.
    """

    provider: Any = None

    def evaluate_case(self, case: EvalCase) -> CaseResult:
        request = case.input.get("request", "")
        deliverable = case.input.get("deliverable", "")

        report = check_deliverable(request, deliverable)
        summary = summarize(report)

        predicted: dict[str, Any] = dict(summary)
        if summary["verdict"] == "cannot_verify" and not summary["failed"]:
            predicted["verdict"] = "cannot_verify"

        expected = case.expected
        correct = self._score(predicted, expected)

        return CaseResult(
            case_id=case.case_id,
            project=case.project,
            predicted=predicted,
            expected=expected,
            label_origin=case.label_origin,
            split=case.split,
            correct=correct,
            notes=report.render(),
        )

    @staticmethod
    def _score(predicted: dict[str, Any], expected: dict[str, Any]) -> bool | None:
        """Score a deliverable case against its expected conditions."""
        checks: list[bool] = []

        if "verdict" in expected:
            checks.append(predicted.get("verdict") == expected["verdict"])

        if "failed" in expected:
            checks.append(predicted.get("failed") == expected["failed"])

        if "failed_gte" in expected:
            checks.append(predicted.get("failed", 0) >= expected["failed_gte"])

        if "requirement_count" in expected:
            checks.append(predicted.get("requirement_count") == expected["requirement_count"])

        if "requirement_count_gte" in expected:
            checks.append(
                predicted.get("requirement_count", 0) >= expected["requirement_count_gte"]
            )

        if "must_detect_failure_containing" in expected:
            needle = expected["must_detect_failure_containing"].lower()
            failures = " ".join(f.get("text", "") for f in predicted.get("failures", []))
            checks.append(needle in failures.lower())

        if not checks:
            return None
        return all(checks)


def build_cases() -> list[EvalCase]:
    """Load the deliverable checker's evaluation cases."""
    from .core.fixtures import load_project_cases

    return load_project_cases("deliverables")


def build_runner(provider: Any = None) -> DeliverableRunner:
    """Factory used by the project registry."""
    return DeliverableRunner(provider)
