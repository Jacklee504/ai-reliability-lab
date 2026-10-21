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


def verify_review_comment(
    comment: ReviewComment,
    snapshot: CodeSnapshot,
) -> ReviewFinding:
    """Check a review comment against static facts in a code snapshot."""
    assertion = extract_assertion(comment.text)
    finding = ReviewFinding(comment=comment, assertion=assertion)

    if comment.file and comment.file not in snapshot.files:
        finding.verdict = Verdict.INSUFFICIENT
        finding.explanation = (
            "the commented file %r is not in the snapshot, so the comment"
            " cannot be checked" % comment.file
        )
        return finding

    file_text = snapshot.file_text(comment.file)
    if not file_text and len(snapshot.files) == 1:
        file_text = next(iter(snapshot.files.values()))

    if not file_text:
        finding.verdict = Verdict.INSUFFICIENT
        finding.explanation = "the commented file is not in the snapshot"
        return finding

    lines = _lines_with_numbers(file_text)
    if comment.line is not None and not (1 <= comment.line <= len(lines)):
        finding.verdict = Verdict.INSUFFICIENT
        finding.explanation = "the cited line is outside the supplied file"
        return finding
    try:
        ast.parse(file_text)
    except SyntaxError:
        finding.verdict = Verdict.UNCERTAIN
        finding.explanation = "the supplied Python file cannot be parsed for a static check"
        return finding

    # No executable outcome: judge from the code, and say how far the
    # code goes.
    if _MISSING_GUARD.search(assertion):
        subject_match = re.search(r"\bon\s+(\w+)\b", assertion, re.IGNORECASE)
        subject = subject_match.group(1) if subject_match else None
        if subject is None:
            finding.verdict = Verdict.UNCERTAIN
            finding.explanation = "the comment does not identify which value needs a guard"
            return finding
        if _guard_present(file_text, subject, comment.line or 1):
            finding.verdict = Verdict.CONTRADICTED
            finding.explanation = (
                "the code contains a guard of the kind the comment says is missing"
            )
        else:
            finding.verdict = Verdict.SUPPORTED
            finding.explanation = "the code contains no such guard"
        start = comment.line or 1
        finding.evidence = [_span_for_lines(
            snapshot.identifier, comment.file, lines, start, min(start + 6, len(lines)),
        )]
        return finding

    if _UNBOUNDED.search(assertion):
        if _unbounded_present(file_text):
            finding.verdict = Verdict.SUPPORTED
            finding.explanation = "the code has a while True loop without a break"
        elif _bound_present(file_text):
            finding.verdict = Verdict.CONTRADICTED
            finding.explanation = "the code has an explicit finite iterator"
        else:
            finding.verdict = Verdict.UNCERTAIN
            finding.explanation = "the code does not clearly show an unbounded construct"
        start = comment.line or 1
        finding.evidence = [_span_for_lines(
            snapshot.identifier, comment.file, lines, start, min(start + 6, len(lines)),
        )]
        return finding

    if _TOCTOU.search(assertion):
        finding.verdict = Verdict.UNCERTAIN
        finding.explanation = (
            "a concurrency claim cannot be settled by reading this function alone;"
            " the caller and the test would be needed"
        )
        start = comment.line or 1
        finding.evidence = [_span_for_lines(
            snapshot.identifier, comment.file, lines, start, min(start + 6, len(lines)),
        )]
        return finding

    if _NULL_ASSERT.search(assertion):
        finding.verdict = Verdict.UNCERTAIN
        finding.explanation = "a null-triggered exception was not executed or proven by this static check"
        start = comment.line or 1
        finding.evidence = [_span_for_lines(
            snapshot.identifier, comment.file, lines, start, min(start + 6, len(lines)),
        )]
        return finding

    finding.verdict = Verdict.UNCERTAIN
    finding.explanation = (
        "no checker is implemented for this kind of assertion, so the comment"
        " is unverified rather than refuted"
    )
    start = comment.line or 1
    finding.evidence = [_span_for_lines(
        snapshot.identifier, comment.file, lines, start, min(start + 3, len(lines)),
    )]
    return finding


def extract_runtime_assertion(comment: str) -> RuntimeAssertion | None:
    """Recognise a runtime claim and the behaviour it predicts.

    Returns ``None`` when the comment makes no runtime claim that a
    harness could settle.
    """
    assertion = extract_assertion(comment)
    if _DIVIDES.search(assertion):
        return RuntimeAssertion(kind="divides_by_zero", expectation=assertion)
    if _RETURNS_NONE.search(assertion):
        return RuntimeAssertion(kind="returns_none", expectation=assertion)
    if _RAISES.search(assertion):
        return RuntimeAssertion(kind="raises", expectation=assertion)
    return None


def verify_runtime_claim(
    comment: ReviewComment,
    snapshot: CodeSnapshot,
    harness: str,
    *,
    timeout: float = 10.0,
) -> ReviewFinding:
    """Check a runtime claim by actually running the reviewer's harness.

    The harness is the reviewer's code. The tool runs it in a separate
    process and compares the *observed* behaviour with what the comment
    predicted. A timeout, a crash in the harness, or a harness that
    reports nothing all produce "unverified", never a verdict.
    """
    assertion = extract_assertion(comment.text)
    finding = ReviewFinding(comment=comment, assertion=assertion)

    runtime = extract_runtime_assertion(comment.text)
    if runtime is None:
        finding.verdict = Verdict.UNCERTAIN
        finding.explanation = (
            "the comment makes no runtime claim that a harness could settle"
        )
        return finding

    if not harness or not harness.strip():
        finding.verdict = Verdict.UNCERTAIN
        finding.explanation = (
            "no executable harness was supplied for this comment, so the"
            " runtime claim is unverified"
        )
        return finding

    outcome = run_harness(snapshot.files, harness, timeout=timeout)
    finding.execution = outcome

    if not outcome.observed:
        finding.verdict = Verdict.UNCERTAIN
        finding.explanation = (
            "the harness did not produce a usable result (%s), so the"
            " runtime claim is unverified" % (outcome.error or "unknown reason")
        )
        return finding

    if runtime.kind == "raises":
        if outcome.raised:
            finding.verdict = Verdict.SUPPORTED
            finding.explanation = (
                "executing the harness raised %s, which is what the comment"
                " predicted" % (outcome.exception_type or "an exception")
            )
        else:
            finding.verdict = Verdict.CONTRADICTED
            finding.explanation = (
                "the harness ran to completion without raising, so the"
                " predicted error did not occur"
            )
        finding.method = Method.EXECUTED
        return finding

    if runtime.kind == "returns_none":
        returned = outcome.extra.get("returned_none")
        if returned is None:
            finding.verdict = Verdict.UNCERTAIN
            finding.explanation = (
                "the harness did not report a return value, so the claim"
                " about the return value is unverified"
            )
            return finding
        if returned:
            finding.verdict = Verdict.SUPPORTED
            finding.explanation = "the harness observed a None return value"
        else:
            finding.verdict = Verdict.CONTRADICTED
            finding.explanation = "the harness observed a non-None return value"
        finding.method = Method.EXECUTED
        return finding

    if runtime.kind == "divides_by_zero":
        if outcome.raised and outcome.exception_type in {
            "ZeroDivisionError",
            "DivisionByZero",
        }:
            finding.verdict = Verdict.SUPPORTED
            finding.explanation = (
                "executing the harness raised %s, which is what the comment"
                " predicted" % outcome.exception_type
            )
        elif outcome.raised:
            finding.verdict = Verdict.UNCERTAIN
            finding.explanation = (
                "the harness raised %s, which is not the predicted"
                " division by zero" % outcome.exception_type
            )
            return finding
        else:
            finding.verdict = Verdict.CONTRADICTED
            finding.explanation = (
                "the harness ran to completion without a division by zero"
            )
        finding.method = Method.EXECUTED
        return finding

    finding.verdict = Verdict.UNCERTAIN
    finding.explanation = "no runtime checker is implemented for this claim"
    return finding


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


@dataclass
class CodeReviewRunner:
    """Runs a code review comment verification case."""

    provider: Any = None
    timeout: float = 10.0

    def evaluate_case(self, case: EvalCase) -> CaseResult:
        inputs = case.input
        snapshot = CodeSnapshot(
            identifier=inputs.get("snapshot_id", "snapshot"),
            files=dict(inputs.get("files", {})),
        )
        comments = [
            ReviewComment(
                text=item.get("text", ""),
                file=item.get("file", ""),
                line=item.get("line"),
            )
            for item in inputs.get("comments", [])
        ]
        # A harness is the reviewer's code for settling a runtime
        # claim. It may be given once for the case, once per comment,
        # or keyed by the comment text.
        case_harness = inputs.get("harness", "")
        findings = []
        for index, comment in enumerate(comments):
            harness = ""
            if isinstance(case_harness, list) and index < len(case_harness):
                harness = case_harness[index]
            elif isinstance(case_harness, dict):
                harness = case_harness.get(comment.text, "")
            elif isinstance(case_harness, str):
                harness = case_harness
            if harness:
                findings.append(
                    verify_runtime_claim(
                        comment, snapshot, harness, timeout=self.timeout
                    )
                )
            else:
                findings.append(verify_review_comment(comment, snapshot))

        expected = case.expected
        if "verdict" in expected:
            wanted = expected["verdict"]
            if isinstance(wanted, list):
                correct = bool(findings) and all(
                    f.verdict.value in wanted for f in findings
                )
            else:
                correct = len(findings) == 1 and findings[0].verdict.value == wanted
        else:
            correct = None

        predicted = {
            "verdict": findings[0].verdict.value if len(findings) == 1 else "multiple",
            "verdicts": [f.verdict.value for f in findings],
            "methods": [f.method.value for f in findings],
        }

        return CaseResult(
            case_id=case.case_id,
            project=case.project,
            predicted=predicted,
            expected=expected,
            label_origin=case.label_origin,
            split=case.split,
            correct=correct,
            notes="\n".join(
                "%s -> %s: %s" % (f.comment.text, f.verdict.value, f.explanation)
                for f in findings
            ),
        )


def build_cases() -> list[EvalCase]:
    """Load the code review verifier evaluation cases."""
    from ..core.fixtures import load_project_cases

    return load_project_cases("code_review")


def build_runner(provider: Any = None) -> CodeReviewRunner:
    """Factory used by the project registry."""
    return CodeReviewRunner(provider=provider)
