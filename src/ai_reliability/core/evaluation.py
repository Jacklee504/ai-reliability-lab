"""Evaluation: cases, runners, metrics, and run comparison.

Every project in the lab exposes an ``evaluate_case`` function with the
same shape, so one runner and one metric layer serves all ten. The
metrics deliberately report **counts first**. With a few dozen cases per
project, a percentage reads like precision it does not have, so the
report shows the numerator next to the denominator and only adds a
derived rate when the denominator is big enough to be meaningful.

Label provenance is carried through to the report. Results built on
synthetic transformations or model-drafted labels are reported in a
separate section from anything human-verified.

Scoring model
-------------
A runner may set ``result.correct`` itself. When it does, that value is
used and :func:`score_case` is not consulted. This lets a project define
its own success criterion (a deliverable case has several conditions,
not one label) while the shared metric layer still works from the raw
counts.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from .types import CaseResult, EvalCase, LabelOrigin, Split

#: Below this many cases, rates are shown as raw counts only.
MIN_CASES_FOR_RATE = 5


class CaseRunner(Protocol):
    """A project implements this to be evaluated."""

    def evaluate_case(self, case: EvalCase) -> CaseResult:
        ...


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


def confusion(predicted: str, expected: str) -> str:
    """Map a (predicted, expected) pair to a confusion cell name."""
    return f"expected={expected},predicted={predicted}"


def score_case(
    predicted: dict[str, Any],
    expected: dict[str, Any],
    *,
    key: str = "verdict",
) -> bool | None:
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


@dataclass
class EvaluationReport:
    """The result of running a project over a set of cases."""

    project: str
    method: str
    results: list[CaseResult] = field(default_factory=list)
    metrics: list[MetricBlock] = field(default_factory=list)
    confusion: dict[str, int] = field(default_factory=dict)
    label_origins: dict[str, int] = field(default_factory=dict)
    splits: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    model_id: str | None = None
    model_calls: int = 0
    parse_failures: int = 0
    total_latency_ms: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def scored(self) -> int:
        return sum(1 for r in self.results if r.correct is not None)

    @property
    def correct(self) -> int:
        return sum(1 for r in self.results if r.correct is True)

    @property
    def human_verified_cases(self) -> int:
        return self.label_origins.get(LabelOrigin.HUMAN_VERIFIED.value, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "method": self.method,
            "model_id": self.model_id,
            "model_calls": self.model_calls,
            "parse_failures": self.parse_failures,
            "total_latency_ms": round(self.total_latency_ms, 2),
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
            f"Project: {self.project}  method: {self.method}",
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


def run_evaluation(
    project: str,
    cases: Iterable[EvalCase],
    runner: CaseRunner,
    *,
    method: str = "deterministic",
    model_id: str | None = None,
    score_key: str = "verdict",
    notes: Iterable[str] = (),
) -> EvaluationReport:
    """Run every case and build an :class:`EvaluationReport`."""
    report = EvaluationReport(project=project, method=method, model_id=model_id)
    confusion_counts: Counter = Counter()
    origins: Counter = Counter()
    splits: Counter = Counter()

    for case in cases:
        result = runner.evaluate_case(case)
        result.expected = dict(case.expected)
        result.label_origin = case.label_origin
        result.split = case.split
        if result.correct is None:
            result.correct = score_case(result.predicted, case.expected, key=score_key)
        report.results.append(result)

        if result.correct is not None:
            confusion_counts[confusion(
                str(result.predicted.get(score_key)),
                str(case.expected.get(score_key)),
            )] += 1
        origins[case.label_origin.value] += 1
        splits[case.split.value] += 1
        report.model_calls += result.model_calls
        report.parse_failures += result.parse_failures
        report.total_latency_ms += result.latency_ms

    report.confusion = dict(confusion_counts)
    report.label_origins = dict(origins)
    report.splits = dict(splits)
    report.notes.extend(notes)
    report.metrics = build_metrics(report)
    if report.label_origins.get(LabelOrigin.HUMAN_VERIFIED.value, 0) != report.total:
        report.notes.append("Labels are not independently human verified; agreement is provisional.")
    if report.splits.get(Split.TEST.value, 0) == 0:
        report.notes.append("No held-out test cases were evaluated.")
    return report


def _expected_verdicts(expected: dict[str, Any]) -> set[str]:
    """Normalise an expected verdict, which may be a list of alternatives."""
    value = expected.get("verdict")
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(v) for v in value}
    return {str(value)}


def build_metrics(report: EvaluationReport) -> list[MetricBlock]:
    """Derive the standard metric set from raw results.

    The false-accusation metric is computed over cases whose expected
    label is a *supported* claim; the false-alarm metric is computed over
    cases whose expected label is a *no-finding* case. Both are reported
    as counts, with a rate only when the denominator is large enough.
    """
    metrics: list[MetricBlock] = []
    scored = [r for r in report.results if r.correct is not None]
    metrics.append(
        MetricBlock(
            name="accepted_outcome",
            numerator=sum(1 for r in scored if r.correct),
            denominator=len(scored),
            note="project case accepted the prediction; some labels allow alternatives",
        )
    )

    # Only cases that expect exactly "supported" count towards the
    # false-accusation denominator. A case whose expected value is a list
    # such as ["supported", "contradicted"] is a multi-outcome case, not
    # a "the source supports this" case, and mixing it in would inflate
    # the false-accusation rate with cases that are not accusations at all.
    supported_cases = [
        r for r in report.results if r.expected.get("verdict") == "supported"
    ]
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
        r for r in report.results
        if r.expected.get("verdict") in ("no_finding", "passed")
    ]
    false_alarms = [
        r for r in no_finding_cases
        if r.predicted.get("verdict") not in {"no_finding", "passed"}
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
