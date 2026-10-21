"""Evaluation: cases, runners, and metrics.

The metrics deliberately report **counts first**. With a few dozen
cases a percentage reads like precision it does not have, so every
metric shows its numerator next to its denominator, and a rate is
only shown when the denominator is big enough to be meaningful.

Label provenance is carried through to the report. A number computed
over model-drafted labels is not an accuracy claim, and the report
says so.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .auditor import SourceDocument, audit_claims
from .providers import Provider
from .types import Verdict

#: Below this many cases, rates are shown as raw counts only.
MIN_CASES_FOR_RATE = 5


class LabelOrigin(str, Enum):
    """Provenance of an expected label in an evaluation case."""

    HUMAN_VERIFIED = "human_verified"
    SYNTHETIC_TRANSFORM = "synthetic_transform"
    MODEL_DRAFTED = "model_drafted"
    UNKNOWN = "unknown"


class Split(str, Enum):
    """Dataset split. ``TRAIN`` must never be tuned on and reported as test."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


@dataclass
class EvalCase:
    """One evaluation case."""

    case_id: str
    input: dict[str, Any]
    expected: dict[str, Any] = field(default_factory=dict)
    label_origin: LabelOrigin = LabelOrigin.UNKNOWN
    split: Split = Split.TEST
    rationale: str = ""
    tags: tuple[str, ...] = ()


@dataclass
class CaseResult:
    """The result of running one case."""

    case_id: str
    predicted: dict[str, Any]
    expected: dict[str, Any] = field(default_factory=dict)
    correct: bool | None = None
    label_origin: LabelOrigin = LabelOrigin.UNKNOWN
    split: Split = Split.TEST
    model_calls: int = 0
    parse_failures: int = 0
    latency_ms: float = 0.0
    notes: str = ""


@dataclass
class MetricBlock:
    """One metric with its numerator and denominator made explicit."""

    name: str
    numerator: int
    denominator: int
    note: str = ""

    @property
    def rate(self) -> float | None:
        if self.denominator == 0 or self.denominator < MIN_CASES_FOR_RATE:
            return None
        return self.numerator / self.denominator

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "rate": None if self.rate is None else round(self.rate, 4),
            "note": self.note,
        }

    def render(self) -> str:
        if self.rate is None:
            return f"{self.name}: {self.numerator}/{self.denominator} (count only)"
        return f"{self.name}: {self.numerator}/{self.denominator} = {self.rate:.3f}"


@dataclass
class EvaluationReport:
    """The result of running the auditor over a set of cases."""

    method: str = "hybrid"
    model_id: str | None = None
    results: list[CaseResult] = field(default_factory=list)
    metrics: list[MetricBlock] = field(default_factory=list)
    confusion: dict[str, int] = field(default_factory=dict)
    label_origins: dict[str, int] = field(default_factory=dict)
    splits: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    model_calls: int = 0
    parse_failures: int = 0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def correct(self) -> int:
        return sum(1 for r in self.results if r.correct is True)

    @property
    def scored(self) -> int:
        return sum(1 for r in self.results if r.correct is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "model_id": self.model_id,
            "model_calls": self.model_calls,
            "parse_failures": self.parse_failures,
            "total_cases": self.total,
            "scored_cases": self.scored,
            "correct_cases": self.correct,
            "label_origins": self.label_origins,
            "splits": self.splits,
            "metrics": [m.to_dict() for m in self.metrics],
            "confusion": self.confusion,
            "notes": self.notes,
            "results": [r.to_dict() for r in self.results],
        }

    def render(self) -> str:
        lines = [
            f"Method: {self.method}",
            f"Model: {self.model_id or 'none (deterministic)'}   "
            f"model calls: {self.model_calls}   parse failures: {self.parse_failures}",
            f"Cases: {self.total} total, {self.scored} scored, {self.correct} correct",
            f"Label origins: {self.label_origins}",
            f"Splits: {self.splits}",
            "Metrics:",
        ]
        for metric in self.metrics:
            lines.append(f"  - {metric.render()}")
        if self.confusion:
            lines.append("Confusion:")
            for cell, count in sorted(self.confusion.items()):
                lines.append(f"  {cell}: {count}")
        for note in self.notes:
            lines.append(f"Note: {note}")
        return "\n".join(lines)


@dataclass
class ClaimRunner:
    """Runs one evaluation case through the auditor."""

    provider: Any = None
    use_model: bool = True

    def evaluate_case(self, case: EvalCase) -> CaseResult:
        inputs = case.input
        answer = inputs.get("answer", "")
        documents = [
            SourceDocument(
                identifier=item.get("identifier", f"source_{index}"),
                text=item.get("text", ""),
                version=item.get("version"),
            )
            for index, item in enumerate(inputs.get("sources", []))
        ]
        report = audit_claims(
            answer, documents, provider=self.provider, use_model=self.use_model
        )
        findings = report.findings
        expected = case.expected

        if "verdicts" in expected:
            correct = [f.verdict.verdict.value for f in findings] == expected["verdicts"]
        elif "verdict" in expected:
            wanted = expected["verdict"]
            if isinstance(wanted, list):
                correct = all(f.verdict.verdict.value in wanted for f in findings)
            else:
                correct = len(findings) == 1 and findings[0].verdict.verdict.value == wanted
        else:
            correct = None

        predicted = {
            "verdict": findings[0].verdict.verdict.value if len(findings) == 1 else "multiple",
            "verdicts": [f.verdict.verdict.value for f in findings],
            "claim_count": len(findings),
        }
        return CaseResult(
            case_id=case.case_id,
            predicted=predicted,
            expected=expected,
            correct=correct,
            label_origin=case.label_origin,
            split=case.split,
            model_calls=report.model_calls,
            parse_failures=report.parse_failures,
            notes=report.render(),
        )


def score_case(predicted: dict[str, Any], expected: dict[str, Any], key: str = "verdict") -> bool | None:
    """Compare a prediction with an expected label.

    Returns ``None`` when the case has no expected value for ``key``,
    which keeps "cannot verify" cases from being scored as wrong.
    """
    if key not in expected:
        return None
    want = expected[key]
    got = predicted.get(key)
    if isinstance(want, list):
        return got in want
    return got == want


def build_metrics(report: EvaluationReport) -> list[MetricBlock]:
    """Derive the standard metric set from raw results."""
    metrics: list[MetricBlock] = []
    scored = [r for r in report.results if r.correct is not None]
    metrics.append(
        MetricBlock(
            name="accepted_outcome",
            numerator=sum(1 for r in scored if r.correct),
            denominator=len(scored),
            note="case accepted the prediction; some labels allow alternatives",
        )
    )
    supported_cases = [r for r in report.results if r.expected.get("verdict") == "supported"]
    false_accusations = [
        r for r in supported_cases if r.predicted.get("verdict") == "contradicted"
    ]
    metrics.append(
        MetricBlock(
            name="false_accusations_on_supported",
            numerator=len(false_accusations),
            denominator=len(supported_cases),
            note="supported claims wrongly called contradicted",
        )
    )
    no_finding_cases = [
        r for r in report.results if r.expected.get("verdict") in ("no_finding", "passed")
    ]
    false_alarms = [
        r for r in no_finding_cases if r.predicted.get("verdict") not in {"no_finding", "passed"}
    ]
    metrics.append(
        MetricBlock(
            name="false_alarms_on_benign",
            numerator=len(false_alarms),
            denominator=len(no_finding_cases),
            note="benign cases wrongly flagged",
        )
    )
    abstained = [
        r for r in report.results
        if r.predicted.get("verdict") in {"insufficient_evidence", "uncertain", "cannot_verify"}
    ]
    metrics.append(
        MetricBlock(
            name="abstentions",
            numerator=len(abstained),
            denominator=len(report.results),
            note="outcomes that declined to assert",
        )
    )
    return metrics


def run_evaluation(
    cases: Iterable[EvalCase],
    runner: ClaimRunner,
    *,
    method: str = "hybrid",
    model_id: str | None = None,
) -> EvaluationReport:
    """Run every case and build an :class:`EvaluationReport`."""
    report = EvaluationReport(method=method, model_id=model_id)
    confusion_counts: Counter = Counter()
    origins: Counter = Counter()
    splits: Counter = Counter()

    for case in cases:
        result = runner.evaluate_case(case)
        result.expected = dict(case.expected)
        result.label_origin = case.label_origin
        result.split = case.split
        if result.correct is None:
            result.correct = score_case(result.predicted, case.expected)
        report.results.append(result)
        if result.correct is not None:
            confusion_counts[
                "expected=%s,predicted=%s"
                % (result.expected.get("verdict"), result.predicted.get("verdict"))
            ] += 1
        origins[case.label_origin.value] += 1
        splits[case.split.value] += 1
        report.model_calls += result.model_calls
        report.parse_failures += result.parse_failures

    report.confusion = dict(confusion_counts)
    report.label_origins = dict(origins)
    report.splits = dict(splits)
    report.metrics = build_metrics(report)
    if report.label_origins.get(LabelOrigin.HUMAN_VERIFIED.value, 0) != report.total:
        report.notes.append("Labels are not independently human verified; agreement is provisional.")
    if report.splits.get(Split.TEST.value, 0) == 0:
        report.notes.append("No held-out test cases were evaluated.")
    return report


def load_cases(path: str | Path) -> list[EvalCase]:
    """Load evaluation cases from a JSON fixture file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("cases", []) if isinstance(payload, dict) else payload
    return [
        EvalCase(
            case_id=entry["case_id"],
            input=entry.get("input", {}),
            expected=entry.get("expected", {}),
            label_origin=LabelOrigin(entry.get("label_origin", "unknown")),
            split=Split(entry.get("split", "test")),
            rationale=entry.get("rationale", ""),
            tags=tuple(entry.get("tags", [])),
        )
        for entry in entries
    ]


def load_all_cases(directory: str | Path) -> list[EvalCase]:
    """Load every case in a fixture directory, across its splits.

    ``cases.json`` holds the development cases and ``test_cases.json``
    holds the held-out set. The split recorded on each case is the one
    it came from, so the report can show which cases were only ever
    scored once.
    """
    base = Path(directory)
    cases = load_cases(base / "cases.json")
    test_file = base / "test_cases.json"
    if test_file.exists():
        for case in load_cases(test_file):
            if case.split is not Split.TEST:
                case = EvalCase(
                    case_id=case.case_id,
                    input=case.input,
                    expected=case.expected,
                    label_origin=case.label_origin,
                    split=Split.TEST,
                    rationale=case.rationale,
                    tags=case.tags,
                )
            cases.append(case)
    return cases


def save_run(report: EvaluationReport, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target
