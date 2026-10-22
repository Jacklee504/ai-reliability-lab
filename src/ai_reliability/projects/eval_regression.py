"""Project 6: evaluation and grounding regression suite.

Problem
-------
AI reliability tools are often demonstrated on a few persuasive examples
without measuring false positives or changes over time. A tool that looks
good on three cases tells you nothing about what happens on case forty.

Approach
--------
This project is the measurement layer for the whole lab. It:

* defines the common case format and the label provenance rules,
* runs every project over one shared case set,
* records raw predictions next to expected labels,
* computes the metrics that matter for a reliability tool, and
* compares two stored runs and lists the cases whose verdict changed.

The metrics that matter here are not accuracy. They are:

false accusation rate
    Supported claims that were called contradicted. A reliability tool
    that accuses wrongly is worse than no tool.

false alarm rate
    Benign cases that were flagged. A detector that flags everything is
    not a detector.

abstention rate
    How often the tool declined to assert. This is a feature, and its
    value depends on the case mix.

Counts come before rates. With tens of cases a percentage reads like
precision it does not have, so every metric reports its numerator and
denominator, and a rate is only shown when the denominator is large
enough to be meaningful.

Deliberate limits
-----------------
* A held-out test split is never used for tuning. The runner reports
  split sizes so it is visible which cases were touched.
* Label provenance is preserved end to end. A number computed over
  model-drafted labels is not an accuracy claim, and the report says so.
* No confidence intervals are computed. The sample sizes do not support
  them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.evaluation import (
    CaseRunner,
    EvaluationReport,
    build_metrics,
    compare_runs,
    run_evaluation,
)
from ..core.fixtures import load_project_cases
from ..core.providers import Provider
from ..core.types import CaseResult, EvalCase, LabelOrigin, Split

#: The projects whose cases the regression suite aggregates.
COVERED_PROJECTS = [
    "claims",
    "citations",
    "omission",
    "deliverables",
    "context_receipts",
    "numerical",
    "instructions",
    "code_review",
    "authorization",
    "retrieval_eval",
]


@dataclass
class RegressionRunner:
    """Runs every project case and aggregates the results."""

    provider: Any = None
    projects: list[str] = field(default_factory=lambda: list(COVERED_PROJECTS))

    def evaluate_case(self, case: EvalCase) -> CaseResult:
        """Run one case through the project that owns it.

        The regression suite reuses each project own runner so a case
        is scored exactly as it would be when that project is run alone.
        """
        from .registry import get_project

        entry = get_project(case.project)
        runner = entry.build_runner(self.provider)
        result = runner.evaluate_case(case)
        result.project = case.project
        return result


def build_cases(projects: list[str] | None = None) -> list[EvalCase]:
    """Collect the cases from every covered project, across all splits.

    Each project may hold a held-out ``test_cases.json`` file. Those
    cases are included here so the regression suite can be run over the
    whole corpus, and can be filtered to a single split with the
    ``--split`` flag.
    """
    cases: list[EvalCase] = []
    for project in projects or COVERED_PROJECTS:
        cases.extend(load_project_cases(project))
    return cases


def build_runner(provider: Any = None) -> RegressionRunner:
    """Factory used by the project registry."""
    return RegressionRunner(provider=provider)


def run_suite(
    provider: Provider | None = None,
    *,
    projects: list[str] | None = None,
    split: Split | None = None,
    method: str = "regression-suite",
) -> EvaluationReport:
    """Run every project case and build the aggregate report.

    The report keeps the per-case predictions and the expected labels
    so an error can be inspected rather than guessed at.
    """
    cases = build_cases(projects)
    if split is not None:
        cases = [c for c in cases if c.split is split]

    runner = RegressionRunner(provider=provider, projects=projects or list(COVERED_PROJECTS))
    report = run_evaluation(
        "eval_regression",
        cases,
        runner,
        method=method,
        model_id=getattr(provider, "model_id", None),
    )
    report.metrics = build_metrics(report)
    report.notes.extend(_provenance_notes(report))
    return report


def _provenance_notes(report: EvaluationReport) -> list[str]:
    """Notes that keep the label provenance visible in the report."""
    notes: list[str] = []
    human = report.label_origins.get(LabelOrigin.HUMAN_VERIFIED.value, 0)
    total = report.total
    if human < total:
        notes.append(
            "%d of %d cases do not carry a human-verified label; the numbers"
            " above are not an accuracy claim for those cases" % (total - human, total)
        )
    else:
        notes.append(
            "all %d cases carry human-verified labels" % total
        )
    if report.splits.get(Split.TRAIN.value):
        notes.append(
            "%d case(s) are in the train split; they must not be used for"
            " tuning and then reported as a test result"
            % report.splits[Split.TRAIN.value]
        )
    if report.splits.get(Split.DEV.value):
        notes.append(
            "%d case(s) are in the dev split" % report.splits[Split.DEV.value]
        )
    return notes


def compare_stored_runs(left_path: str, right_path: str) -> dict[str, Any]:
    """Compare two stored runs and list the cases whose verdict changed."""
    from ..core.evaluation import load_run

    return compare_runs(load_run(left_path), load_run(right_path))


@dataclass
class BaselineRunner:
    """A deliberately simple baseline for comparison.

    The baseline answers the easiest question that is still meaningful:
    "is the claim supported by the source at all?". It looks for the
    claim distinctive terms in the source and answers supported when
    they are there.

    A method that cannot beat this on the false-accusation metric is not
    earning its complexity.
    """

    provider: Any = None
    always: str = "supported"

    def evaluate_case(self, case: EvalCase) -> CaseResult:
        from ..core.documents import SourceDocument
        from .claims import audit_claims

        inputs = case.input
        answer = inputs.get("answer", "")
        documents = [
            SourceDocument(
                identifier=item.get("identifier", "source"),
                text=item.get("text", ""),
            )
            for item in inputs.get("sources", [])
        ]
        if documents:
            report = audit_claims(answer, documents, provider=None, use_model=False)
            verdicts = [f.verdict.verdict.value for f in report.findings]
            predicted = {
                "verdict": verdicts[0] if len(verdicts) == 1 else "multiple",
                "verdicts": verdicts,
            }
        else:
            predicted = {"verdict": self.always, "verdicts": [self.always]}

        expected = case.expected
        wanted = expected.get("verdict")
        if isinstance(wanted, list):
            correct = predicted.get("verdict") in wanted
        else:
            correct = predicted.get("verdict") == wanted

        return CaseResult(
            case_id=case.case_id,
            project=case.project,
            predicted=predicted,
            expected=expected,
            label_origin=case.label_origin,
            split=case.split,
            correct=correct,
            notes="baseline: " + str(predicted.get("verdict")),
        )


def run_baseline(
    provider: Provider | None = None,
    *,
    projects: list[str] | None = None,
    split: Split | None = None,
) -> EvaluationReport:
    """Run the simple baseline over the same cases."""
    cases = build_cases(projects)
    if split is not None:
        cases = [c for c in cases if c.split is split]

    report = run_evaluation(
        "eval_regression_baseline",
        cases,
        BaselineRunner(provider=provider),
        method="keyword-overlap-baseline",
        model_id=None,
    )
    report.metrics = build_metrics(report)
    report.notes.extend(_provenance_notes(report))
    return report
