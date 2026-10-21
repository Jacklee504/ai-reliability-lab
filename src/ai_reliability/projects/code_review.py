"""Project 9: AI code-review comment verifier.

Problem
-------
An AI review comment can sound precise while being technically false. It
cites a real line number, quotes real code, and still ignores a guard, a
type constraint, a caller, or a test elsewhere in the file.

Approach
--------
The verifier extracts the **checkable assertion** from a comment, then
looks for evidence in the code. It only asserts facts that its static
checks can establish and abstains on runtime behavior.

Three outcomes are reported:

* **supported** - the code bears the comment out.
* **refuted** - the code contradicts the comment.
* **unverified** - the code does not settle it.

The distinction between refuted and unverified matters. A review comment
about a race condition cannot be refuted by reading the function in
isolation; it can only be unverified without the caller.

Deliberate limits
-----------------
* The verifier never changes code. It reports a verdict and the smallest
  relevant span, and stops.
* It does not evaluate style or taste. A comment that is wrong about
  style is not a technical falsehood.
* Evidence is quoted exactly, with file and line numbers.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

from ..core.execution import ExecutionOutcome, run_harness
from ..core.types import CaseResult, EvalCase, Method, Span, Verdict, short_id

#: Assertion shapes the verifier recognises.
_NULL_ASSERT = re.compile(
    r"\b(?:will|would|must|can|should)\s+(?:not\s+)?(?:\w+\s+){0,3}?" + r"(?:raise|crash|error|overflow|none|null|nil)\b",
    re.IGNORECASE,
)
_ALWAYS_TRUE = re.compile(
    r"\b(?:always|never|every time|guaranteed|cannot|must)\b",
    re.IGNORECASE,
)
_TYPE_ASSERT = re.compile(
    r"\b(?:returns?|is)\s+(?:always\s+)?(?:an?\s+)?(\w+)\b",
    re.IGNORECASE,
)
_UNBOUNDED = re.compile(
    r"\b(?:unbounded|unlimited|infinite|no limit|without bound)\b",
    re.IGNORECASE,
)
_TOCTOU = re.compile(
    r"\b(?:race condition|toctou|time of check|time of use)\b",
    re.IGNORECASE,
)
_MISSING_GUARD = re.compile(
    r"\b(?:no|missing|without|lacks?)\s+(?:a\s+|an\s+|the\s+)?" + r"(?:null|none|empty|existence|input|validation|error)\s*" + r"(?:check|guard|validation)?\b",
    re.IGNORECASE,
)
_OFF_BY_ONE = re.compile(
    r"\boff[- ]by[- ]one\b",
    re.IGNORECASE,
)

#: Comments that make a claim about runtime behaviour. These are the
#: ones a static reading cannot settle, so they are the ones a harness
#: can settle.
_RAISES = re.compile(
    r"\b(?:will|would|can|might|should)\s+(?:not\s+)?(?:raise|throw|error|crash)\b",
    re.IGNORECASE,
)
_RETURNS_NONE = re.compile(
    r"\breturns?\s+(?:always\s+)?none\b",
    re.IGNORECASE,
)
_DIVIDES = re.compile(
    r"\b(?:divide|division|divides)\b.*\bzero\b|\bzero[- ]division\b",
    re.IGNORECASE,
)


@dataclass
class RuntimeAssertion:
    """A runtime claim, and the harness that can settle it.

    ``expectation`` is what the comment says will happen. The harness
    is the reviewer's code; the tool only decides whether the observed
    behaviour matches the claim.
    """

    kind: str  # "raises" | "returns_none" | "divides_by_zero"
    expectation: str
    harness: str = ""

@dataclass
class CodeSnapshot:
    """A synthetic code snapshot the comment is checked against."""

    identifier: str
    files: dict[str, str] = field(default_factory=dict)

    def file_text(self, path: str) -> str:
        return self.files.get(path, "")


@dataclass
class ReviewComment:
    """A review comment and the location it points at."""

    text: str
    file: str = ""
    line: int | None = None


@dataclass
class ReviewFinding:
    """The verdict on one review comment."""

    comment: ReviewComment
    assertion: str = ""
    verdict: Verdict = Verdict.UNCERTAIN
    evidence: list[Span] = field(default_factory=list)
    explanation: str = ""
    method: Method = Method.DETERMINISTIC
    execution: ExecutionOutcome | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "comment": {
                "text": self.comment.text,
                "file": self.comment.file,
                "line": self.comment.line,
            },
            "assertion": self.assertion,
            "verdict": self.verdict.value,
            "explanation": self.explanation,
            "evidence": [s.to_dict() for s in self.evidence],
            "method": self.method.value,
            "execution": self.execution.to_dict() if self.execution else None,
        }


def extract_assertion(comment: str) -> str:
    """Extract the checkable technical assertion from a review comment.

    The assertion is the part of the comment that could be true or false
    about the code. Preamble and advice are dropped.
    """
    text = comment.strip()
    text = re.sub(
        r"^(?:\s*(?:nit|style|note|warning|error|bug|issue|blocker|comment)\s*[:\-]\s*)+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    match = re.search(r"([^.]+)\.", text)
    if match:
        text = match.group(1)
    return text.strip()


def _lines_with_numbers(text: str) -> list[tuple[int, str]]:
    """Return (line_number, line) for each line of a file, 1-based."""
    return [(index, line) for index, line in enumerate(text.splitlines(), start=1)]


def _span_for_lines(
    identifier: str,
    path: str,
    lines: list[tuple[int, str]],
    start: int,
    end: int,
    note: str = "",
) -> Span:
    """Build a span covering a line range of a file."""
    chosen = [(number, text) for number, text in lines if start <= number <= end]
    body = "\n".join(text for _number, text in chosen)
    return Span(
        span_id=short_id("sp", identifier, path, str(start), str(end)),
        source_id=identifier,
        text=body,
        start=0,
        end=len(body),
        section="%s:%d-%d" % (path, start, end),
        warning=note or None,
        kind="code",
    )
def _guard_present(text: str, subject: str | None = None, line: int = 1) -> bool:
    """Find an early null guard in the function containing the cited line."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.lineno <= line <= node.end_lineno
    ]
    if not functions:
        return False
    function = min(functions, key=lambda node: node.end_lineno - node.lineno)
    operations = [
        child.lineno for child in ast.walk(function)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
        and child.func.attr in {"execute", "query"}
    ]
    first_operation = min(operations, default=function.end_lineno + 1)
    for node in function.body:
        if not isinstance(node, ast.If):
            continue
        if node.lineno >= first_operation:
            continue
        compared = node.test
        if not isinstance(compared, ast.Compare) or len(compared.ops) != 1:
            continue
        if not isinstance(compared.ops[0], (ast.Is, ast.Eq)):
            continue
        if len(compared.comparators) != 1:
            continue
        left, right = compared.left, compared.comparators[0]
        name = left if isinstance(right, ast.Constant) and right.value is None else right
        other = right if name is left else left
        if not isinstance(name, ast.Name) or not isinstance(other, ast.Constant) or other.value is not None:
            continue
        if subject and name.id != subject:
            continue
        if any(isinstance(child, (ast.Raise, ast.Return)) for child in node.body):
            return True
    return False


def _unbounded_present(text: str) -> bool:
    """Find a syntactically unbounded while loop without a break."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value is True:
            if not any(isinstance(child, ast.Break) for child in ast.walk(node)):
                return True
    return False


def _bound_present(text: str) -> bool:
    """Find a for loop over an explicitly finite literal or range."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        if isinstance(node.iter, (ast.List, ast.Tuple, ast.Set)):
            return True
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
            return True
    return False
