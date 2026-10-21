"""Project 8: coding-agent instruction compliance tester.

Problem
-------
A coding agent may receive repository instructions and overlook a
measurable requirement. The output looks finished, so the omission is
only visible if you re-read the instructions against the diff.

Approach
--------
Instructions are parsed into explicit **predicates**. A predicate says
what must be true of the final artifacts, and whether it can be checked
mechanically or needs judgement.

Several kinds of predicate are supported:

``file_unchanged``
    "do not modify this file" - the file content hash must be the
    same before and after.

``file_exists`` / ``file_contains``
    "add a test for this case", "keep this output format" - the file
    must exist and contain a required string or pattern.

``command_run``
    "run the linter" - the agent trace must show the command.

``file_deleted``
    "remove the temporary file" - the file must be gone afterwards.

``file_not_created``
    "do not add new dependencies" - no file matching a pattern may
    appear in the final state that was not in the starting state.

``line_limit``
    "keep this file under 200 lines" - the final file must not exceed
    the stated maximum.

``no_artifact_contains``
    "do not commit secrets" - no final artifact may contain a
    forbidden pattern.

An instruction that cannot be turned into a predicate is reported as
**inconclusive**, never as a pass.

Deliberate limits
-----------------
* The tester checks *artifacts*, not attention. It never claims that an
  agent read an instruction, only that the artifacts satisfy it.
* An ambiguous instruction is reported as inconclusive. Guessing would
  manufacture compliance.
* The fixtures are synthetic miniature repositories. The workspace own
  safety instructions are never altered or used as a fixture.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from ..core.types import CaseResult, CheckStatus, EvalCase, Method, Verdict

#: Instruction forms that the parser recognises.
_UNCHANGED = re.compile(
    r"\b(?:do\s+not|don't|never|avoid)\s+(?:modify|change|edit|touch|alter)\s+"
    r"([\w./-]+\.\w+)",
    re.IGNORECASE,
)
_MUST_EXIST = re.compile(
    r"\b(?:add|create|write|include)\s+(?:a\s+|an\s+|the\s+)?(?:test\s+for\s+)?"
    r"([\w./-]+\.\w+)",
    re.IGNORECASE,
)
_CONTAINS = re.compile(
    r"\b(?:keep|maintain|preserve|retain)\s+(?:the\s+)?output\s+format\s+"
    r"[\"']?([^\"'\n.]+)[\"']?",
    re.IGNORECASE,
)
#: A command to run. The target is the tool word itself, so that a trace
#: entry such as "ruff check ." satisfies "run the linter" only when the
#: trace mentions a linter. The mapping from the instruction word to a
#: trace keyword is handled by :func:`_command_matches`.
_RUN_COMMAND = re.compile(
    r"\b(?:run|execute)\s+(?:the\s+)?([\w -]+?)\b",
    re.IGNORECASE,
)

#: Words that name a check, and the trace keywords that satisfy them.
_COMMAND_ALIASES = {
    "linter": ("lint", "ruff", "flake8", "pylint", "eslint"),
    "lint": ("lint", "ruff", "flake8", "pylint", "eslint"),
    "tests": ("pytest", "test", "unittest", "jest"),
    "test": ("pytest", "test", "unittest", "jest"),
    "type checker": ("mypy", "pyright", "tsc"),
    "typecheck": ("mypy", "pyright", "tsc"),
    "formatter": ("fmt", "format", "black", "prettier"),
    "build": ("build", "make", "compile"),
    "checks": ("check", "lint", "test"),
}
_ADD_TEST_FOR = re.compile(
    r"\badd\s+a\s+test\s+for\s+([\w./-]+\.\w+)",
    re.IGNORECASE,
)

#: "remove the temporary file" - the file must not survive.
_REMOVE_FILE = re.compile(
    r"\b(?:remove|delete|drop|discard)\s+(?:the\s+)?(?:file\s+)?([\w./-]+\.\w+)",
    re.IGNORECASE,
)

#: "do not add new dependencies" - no new file may match a pattern.
_NO_NEW_FILE = re.compile(
    r"\b(?:do\s+not|don't|never|avoid)\s+(?:add|create|introduce)\s+"
    r"(?:any\s+|new\s+|additional\s+)?([\w./-]*\*?[\w.-]*)\s*$",
    re.IGNORECASE,
)

#: "keep this file under 200 lines" - a maximum line count.
_MAX_LINES = re.compile(
    r"\b(?:keep|limit|under|below|at\s+most|no\s+more\s+than|max(?:imum)?)\b"
    r"[^.]{0,80}?\b(\d+)\s+lines?\b",
    re.IGNORECASE,
)
_MAX_LINES_FILE = re.compile(
    r"\b(?:file|module|script)\s+([\w./-]+\.\w+)",
    re.IGNORECASE,
)

#: "do not commit secrets" - a forbidden pattern must not appear.
_NO_FORBIDDEN = re.compile(
    r"\b(?:do\s+not|don't|never|avoid|must\s+not)\s+"
    r"(?:commit|include|add|hard[- ]?code|embed|leak|expose|print|log)?\s*"
    r"(?:any\s+)?(?:hard[- ]?coded\s+)?"
    r"(?:api[_\- ]?keys?|secrets?|credentials?|passwords?|tokens?|private[_\- ]?keys?)\b",
    re.IGNORECASE,
)

#: Patterns that count as a leaked secret when the instruction forbids them.
_SECRET_PATTERNS = [
    re.compile(r"\b(?:api[_\- ]?key|secret[_\- ]?key|access[_\- ]?token)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\bpassword\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]


@dataclass
class Predicate:
    """One checkable requirement derived from an instruction."""

    predicate_id: str
    kind: str  # "file_unchanged" | "file_exists" | "file_contains" | "command_run"
    target: str = ""
    value: str = ""
    check: str = "deterministic"  # "deterministic" | "semantic"
    source_quote: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate_id": self.predicate_id,
            "kind": self.kind,
            "target": self.target,
            "value": self.value,
            "check": self.check,
            "source_quote": self.source_quote,
        }


@dataclass
class PredicateResult:
    """The outcome of evaluating one predicate."""

    predicate: Predicate
    status: CheckStatus
    evidence: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate.to_dict(),
            "status": self.status.value,
            "evidence": self.evidence,
            "detail": self.detail,
        }


@dataclass
class ComplianceReport:
    """The result of replaying an agent output against instructions."""

    predicates: list[Predicate] = field(default_factory=list)
    results: list[PredicateResult] = field(default_factory=list)
    unparsed_instructions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def violated(self) -> list[PredicateResult]:
        return [r for r in self.results if r.status is CheckStatus.FAILED]

    @property
    def satisfied(self) -> list[PredicateResult]:
        return [r for r in self.results if r.status is CheckStatus.PASSED]

    @property
    def inconclusive(self) -> list[PredicateResult]:
        return [r for r in self.results if r.status is CheckStatus.CANNOT_VERIFY]

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate_count": len(self.predicates),
            "passed": len(self.satisfied),
            "failed": len(self.violated),
            "cannot_verify": len(self.inconclusive),
            "violations": [
                {
                    "predicate": r.predicate.to_dict(),
                    "evidence": r.evidence,
                    "detail": r.detail,
                }
                for r in self.violated
            ],
            "results": [r.to_dict() for r in self.results],
            "unparsed_instructions": self.unparsed_instructions,
            "notes": self.notes,
        }

    def render(self) -> str:
        lines = [
            f"Instruction compliance: {len(self.predicates)} predicate(s)",
            f"  passed={len(self.satisfied)} failed={len(self.violated)}"
            f" cannot_verify={len(self.inconclusive)}",
        ]
        for result in self.results:
            marker = {
                CheckStatus.PASSED: "PASS",
                CheckStatus.FAILED: "VIOLATED",
                CheckStatus.CANNOT_VERIFY: "INCONCLUSIVE",
            }[result.status]
            lines.append(f"  [{marker}] {result.predicate.source_quote}")
            if result.status is not CheckStatus.PASSED:
                lines.append(f"      {result.detail}")
        for instruction in self.unparsed_instructions:
            lines.append(f"  [inconclusive instruction] {instruction}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)

def parse_instructions(instructions: list[str]) -> tuple[list[Predicate], list[str]]:
    """Turn repository instructions into predicates.

    Returns (predicates, unparsed). An instruction that no pattern
    recognises is returned in ``unparsed`` so the caller can report it as
    inconclusive rather than dropping it silently.
    """
    predicates: list[Predicate] = []
    unparsed: list[str] = []
    counter = 0

    def add(**kwargs: Any) -> None:
        nonlocal counter
        counter += 1
        kwargs["predicate_id"] = "pred_%02d" % counter
        predicates.append(Predicate(**kwargs))

    for instruction in instructions:
        text = instruction.strip()
        if not text:
            continue
        matched = False

        match = _REMOVE_FILE.search(text)
        if match:
            add(kind="file_deleted", target=match.group(1), source_quote=text)
            matched = True

        match = _UNCHANGED.search(text)
        if match:
            add(kind="file_unchanged", target=match.group(1), source_quote=text)
            matched = True

        match = _MAX_LINES.search(text)
        if match:
            file_match = _MAX_LINES_FILE.search(text)
            add(
                kind="line_limit",
                target=file_match.group(1) if file_match else "",
                value=match.group(1),
                source_quote=text,
            )
            matched = True

        match = _NO_FORBIDDEN.search(text)
        if match:
            add(kind="no_artifact_contains", value="secret", source_quote=text)
            matched = True

        match = _ADD_TEST_FOR.search(text)
        if match:
            add(kind="file_exists", target=match.group(1), source_quote=text)
            matched = True

        match = _CONTAINS.search(text)
        if match:
            add(kind="file_contains", value=match.group(1).strip(), source_quote=text)
            matched = True

        match = _RUN_COMMAND.search(text)
        if match:
            add(kind="command_run", target=match.group(1).strip(), source_quote=text)
            matched = True

        if not matched:
            match = _MUST_EXIST.search(text)
            if match:
                add(kind="file_exists", target=match.group(1), source_quote=text)
                matched = True

        if not matched:
            match = _NO_NEW_FILE.search(text)
            if match:
                add(kind="file_not_created", target=match.group(1), source_quote=text)
                matched = True

        if not matched:
            unparsed.append(text)

    return predicates, unparsed


def _command_matches(target: str, command: str) -> bool:
    """Does a trace command satisfy a command instruction?

    The instruction names a tool in prose ("the linter") and the trace
    names the actual command ("ruff check ."). The matcher maps the
    instruction word to the keywords that would satisfy it, and falls
    back to a literal substring match.
    """
    lowered_target = target.strip().lower()
    lowered_command = command.lower()
    for alias, keywords in _COMMAND_ALIASES.items():
        if alias in lowered_target:
            return any(keyword in lowered_command for keyword in keywords)
    return lowered_target in lowered_command


def _match_path(pattern: str, paths: list[str]) -> str | None:
    """Return the first path matching a glob-style pattern.

    A pattern with a wildcard is matched against the file name and
    against the whole path, so "requirements*.txt" matches
    "docs/requirements_extra.txt" and "*requirements*" matches
    anything with that substring. A pattern with no wildcard is
    matched as a path suffix, which is how a human reads
    "do not add requirements.txt".
    """
    if not pattern:
        return None
    if "*" in pattern or "?" in pattern:
        body = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
        name_regex = re.compile("^" + body + "$", re.IGNORECASE)
        path_regex = re.compile(body, re.IGNORECASE)
    else:
        name_regex = None
        path_regex = re.compile(r"(?:^|/)" + re.escape(pattern) + r"$", re.IGNORECASE)
    for path in paths:
        name = path.rsplit("/", 1)[-1]
        if name_regex is not None:
            if name_regex.search(name) or path_regex.search(path):
                return path
        elif path_regex.search(path):
            return path
    return None


def _hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evaluate_predicates(
    predicates: list[Predicate],
    before: dict[str, str],
    after: dict[str, str],
    trace: list[str] | None = None,
) -> list[PredicateResult]:
    """Evaluate predicates against the before and after file states.

    ``before`` and ``after`` map a path to its content. ``trace`` is the
    list of commands the agent ran, used by the ``command_run`` predicate.
    """
    results: list[PredicateResult] = []
    commands = trace or []

    for predicate in predicates:
        if predicate.kind == "file_unchanged":
            if predicate.target not in after:
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.CANNOT_VERIFY,
                    detail="%s is not present in the final artifacts" % predicate.target,
                ))
                continue
            if predicate.target not in before:
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.CANNOT_VERIFY,
                    detail="%s was not in the starting state, so it cannot be"
                           " shown to be unchanged" % predicate.target,
                ))
                continue
            if _hash_content(before[predicate.target]) == _hash_content(after[predicate.target]):
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.PASSED,
                    evidence="%s content hash is unchanged" % predicate.target,
                ))
            else:
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.FAILED,
                    evidence="%s differs between the starting and final states"
                           % predicate.target,
                    detail="%s was modified even though the instruction said"
                           " not to modify it" % predicate.target,
                ))
            continue

        if predicate.kind == "file_exists":
            if predicate.target in after and after[predicate.target].strip():
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.PASSED,
                    evidence="%s exists and is not empty" % predicate.target,
                ))
            else:
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.FAILED,
                    evidence="%s is absent or empty" % predicate.target,
                    detail="the instruction required %s and it is not in the"
                           " final artifacts" % predicate.target,
                ))
            continue

        if predicate.kind == "file_contains":
            blob = "\n".join(after.values())
            if predicate.value and predicate.value.lower() in blob.lower():
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.PASSED,
                    evidence="%r appears in the final artifacts" % predicate.value,
                ))
            else:
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.FAILED,
                    evidence="%r does not appear in the final artifacts" % predicate.value,
                    detail="the required output format is missing",
                ))
            continue

        if predicate.kind == "file_deleted":
            if predicate.target in after:
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.FAILED,
                    evidence="%s is still present in the final artifacts" % predicate.target,
                    detail="the instruction required %s to be removed" % predicate.target,
                ))
            else:
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.PASSED,
                    evidence="%s is absent from the final artifacts" % predicate.target,
                ))
            continue

        if predicate.kind == "file_not_created":
            pattern = predicate.target
            if not pattern:
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.CANNOT_VERIFY,
                    detail="the instruction does not name a file pattern to forbid",
                ))
                continue
            new_files = sorted(set(after) - set(before))
            matched = _match_path(pattern, new_files)
            if matched:
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.FAILED,
                    evidence="new file %s matches the forbidden pattern %r" % (matched, pattern),
                    detail="the instruction forbade adding this file",
                ))
            else:
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.PASSED,
                    evidence="no newly created file matches %r" % pattern,
                ))
            continue

        if predicate.kind == "line_limit":
            limit = int(predicate.value) if predicate.value.isdigit() else 0
            if limit <= 0:
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.CANNOT_VERIFY,
                    detail="the instruction does not state a usable line limit",
                ))
                continue
            targets = (
                [predicate.target]
                if predicate.target and predicate.target in after
                else sorted(after)
            )
            over = [
                (path, len(after[path].splitlines()))
                for path in targets
                if len(after[path].splitlines()) > limit
            ]
            if over:
                path, count = over[0]
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.FAILED,
                    evidence="%s has %d lines, which is over the limit of %d" % (path, count, limit),
                    detail="the line limit was exceeded",
                ))
            else:
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.PASSED,
                    evidence="no file exceeds the limit of %d lines" % limit,
                ))
            continue

        if predicate.kind == "no_artifact_contains":
            hits: list[tuple[str, str]] = []
            for path, content in sorted(after.items()):
                for secret_re in _SECRET_PATTERNS:
                    match = secret_re.search(content)
                    if match:
                        hits.append((path, match.group(0)[:40]))
                        break
            if hits:
                path, sample = hits[0]
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.FAILED,
                    evidence="%s contains a value matching the forbidden pattern: %s" % (path, sample),
                    detail="the instruction forbade including this kind of value",
                ))
            else:
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.PASSED,
                    evidence="no artifact contains a value matching the forbidden patterns",
                ))
            continue

        if predicate.kind == "command_run":
            hit = next((c for c in commands if _command_matches(predicate.target, c)), None)
            if hit:
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.PASSED,
                    evidence="trace contains %r" % hit,
                ))
            else:
                results.append(PredicateResult(
                    predicate=predicate,
                    status=CheckStatus.FAILED,
                    evidence="no command matching %r in the trace" % predicate.target,
                    detail="the required command does not appear in the agent trace",
                ))
            continue

        results.append(PredicateResult(
            predicate=predicate,
            status=CheckStatus.CANNOT_VERIFY,
            detail="no checker is implemented for this predicate kind",
        ))

    return results


def check_compliance(
    instructions: list[str],
    before: dict[str, str],
    after: dict[str, str],
    trace: list[str] | None = None,
) -> ComplianceReport:
    """Replay a candidate output against a set of instructions."""
    predicates, unparsed = parse_instructions(instructions)
    results = evaluate_predicates(predicates, before, after, trace)

    notes: list[str] = []
    if unparsed:
        notes.append(
            "%d instruction(s) could not be turned into a predicate and are"
            " reported as inconclusive" % len(unparsed)
        )
    notes.append(
        "this checks the artifacts, not the agent attention; a satisfied"
        " predicate does not prove the instruction was read"
    )
    return ComplianceReport(
        predicates=predicates,
        results=results,
        unparsed_instructions=unparsed,
        notes=notes,
    )

# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


@dataclass
class InstructionRunner:
    """Runs an instruction compliance evaluation case."""

    provider: Any = None

    def evaluate_case(self, case: EvalCase) -> CaseResult:
        report = check_compliance(
            list(case.input.get("instructions", [])),
            dict(case.input.get("before", {})),
            dict(case.input.get("after", {})),
            list(case.input.get("trace", [])),
        )

        expected = case.expected
        checks: list[bool] = []

        if "violated" in expected:
            checks.append(len(report.violated) == expected["violated"])
        if "violated_gte" in expected:
            checks.append(len(report.violated) >= expected["violated_gte"])
        if "passed" in expected:
            checks.append(len(report.satisfied) == expected["passed"])
        if "inconclusive_instructions" in expected:
            checks.append(
                len(report.unparsed_instructions) == expected["inconclusive_instructions"]
            )
        if "must_violate_containing" in expected:
            blob = " ".join(
                v["predicate"]["source_quote"] for v in report.to_dict()["violations"]
            ).lower()
            for needle in expected["must_violate_containing"]:
                checks.append(needle.lower() in blob)
        if "verdict" in expected:
            actual = "violated" if report.violated else (
                "inconclusive" if report.inconclusive or report.unparsed_instructions
                else "passed"
            )
            checks.append(actual == expected["verdict"])

        correct = all(checks) if checks else None

        predicted = {
            "verdict": "violated" if report.violated else (
                "inconclusive" if report.inconclusive or report.unparsed_instructions
                else "passed"
            ),
            "violated": len(report.violated),
            "passed": len(report.satisfied),
            "cannot_verify": len(report.inconclusive),
            "unparsed_instruction_count": len(report.unparsed_instructions),
        }

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


def build_cases() -> list[EvalCase]:
    """Load the instruction compliance evaluation cases."""
    from ..core.fixtures import load_project_cases

    return load_project_cases("instructions")


def build_runner(provider: Any = None) -> InstructionRunner:
    """Factory used by the project registry."""
    return InstructionRunner(provider=provider)
