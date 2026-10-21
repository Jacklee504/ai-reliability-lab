"""Registry of the ten project modules.

Each project exposes:

``build_cases()``
    The evaluation cases for that project.

``build_runner(provider)``
    A case runner for that project.

The registry is what ``airl projects`` and ``airl audit eval`` use. Project
modules are imported lazily, so a project that is not yet built is
reported as ``not_started`` instead of breaking the whole CLI.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from ..core.evaluation import CaseRunner, EvaluationReport, run_evaluation
from ..core.fixtures import load_cases
from ..core.providers import Provider
from ..core.types import EvalCase, Split

#: Canonical project order, matching the brief.
PROJECT_ORDER = [
    "omission",
    "claims",
    "citations",
    "deliverables",
    "context_receipts",
    "eval_regression",
    "numerical",
    "instructions",
    "code_review",
    "authorization",
    "source_versions",
    "study_coverage",
    "retrieval_eval",
    "baselines",
]

TITLES = {
    "omission": "Important-omission auditor",
    "claims": "Claim-level source auditor",
    "citations": "Citation support auditor",
    "deliverables": "AI deliverable contract checker",
    "context_receipts": "File-context receipt",
    "eval_regression": "Evaluation and grounding regression suite",
    "numerical": "Numerical and table-claim verifier",
    "instructions": "Coding-agent instruction compliance tester",
    "code_review": "AI code-review comment verifier",
    "authorization": "Agent action authorization checker",
    "source_versions": "Source-version claim tracker",
    "study_coverage": "Study-coverage auditor",
    "retrieval_eval": "Retrieval method comparison (lexical vs embedding)",
    "baselines": "Naive LLM-judge baselines",
}

DESCRIPTIONS = {
    "omission": "Finds source material that a summary left out, given the user's purpose.",
    "claims": "Splits an answer into atomic claims and checks each against supplied sources.",
    "citations": "Checks whether a cited passage supports the sentence it is attached to.",
    "deliverables": "Turns a user request into explicit requirement records and checks the output.",
    "context_receipts": "Records exactly which file version and extracted text were submitted.",
    "eval_regression": "Cross-project runner with splits, label provenance, baselines, and run comparison.",
    "numerical": "Checks a numeric claim against the right row, column, year, and unit.",
    "instructions": "Replays a candidate diff against repository instructions and checks each predicate.",
    "code_review": "Checks whether a review comment's technical assertion holds in the code.",
    "authorization": "Dry-run policy check: was the agent authorised to take this action?",
    "source_versions": "Lists claims whose cited evidence overlaps an edited source region.",
    "study_coverage": "Counts explicit syllabus-topic mentions without asserting learning quality.",
    "retrieval_eval": "Measures whether the ranking stage retrieves the passage that answers the query.",
    "baselines": "Stripped-down LLM judges used to test whether the tool prompts earn their keep.",
}

#: Default method label per project.
METHODS = {
    "baselines": "naive_llm_judge",
    "omission": "hybrid",
    "claims": "hybrid",
    "citations": "hybrid",
    "deliverables": "deterministic",
    "context_receipts": "deterministic",
    "eval_regression": "regression_suite",
    "numerical": "deterministic",
    "instructions": "deterministic",
    "code_review": "deterministic",
    "authorization": "deterministic",
    "source_versions": "deterministic",
    "study_coverage": "deterministic",
    "retrieval_eval": "retrieval_comparison",
}


@dataclass
class ProjectEntry:
    """Everything the registry knows about one project."""

    name: str
    title: str
    status: str
    description: str
    module_name: str
    method: str = "deterministic"

    def _module(self):
        return importlib.import_module(f".{self.module_name}", package=__package__)

    def build_cases(self) -> list[EvalCase]:
        """Load the project's evaluation cases from its fixtures."""
        try:
            module = self._module()
        except ImportError:
            return []
        builder = getattr(module, "build_cases", None)
        if callable(builder):
            return builder()
        return load_cases(self.name)

    def build_runner(self, provider: Provider) -> CaseRunner:
        module = self._module()
        factory = getattr(module, "build_runner", None)
        if not callable(factory):
            raise NotImplementedError(f"project {self.name} does not expose build_runner()")
        return factory(provider)

    def run(self, provider: Provider, split: Split | None = None) -> EvaluationReport:
        cases = self.build_cases()
        if split is not None:
            cases = [c for c in cases if c.split is split]
        runner = self.build_runner(provider)
        return run_evaluation(
            self.name,
            cases,
            runner,
            method=self.method,
            model_id=getattr(provider, "model_id", None),
        )


def registry() -> dict[str, ProjectEntry]:
    """Build the project registry, marking unimplemented projects."""
    entries: dict[str, ProjectEntry] = {}
    for name in PROJECT_ORDER:
        try:
            importlib.import_module(f".{name}", package=__package__)
            status = "runnable"
        except ImportError:
            status = "not_started"
        entries[name] = ProjectEntry(
            name=name,
            title=TITLES[name],
            status=status,
            description=DESCRIPTIONS[name],
            module_name=name,
            method=METHODS[name],
        )
    return entries


def get_project(name: str) -> ProjectEntry:
    entries = registry()
    if name not in entries:
        raise KeyError(f"unknown project {name!r}; known: {sorted(entries)}")
    return entries[name]


def run_project_evaluation(
    name: str,
    provider: Provider,
    split: Split | None = None,
    method: str | None = None,
    baseline_kind: str = "naive_binary",
) -> EvaluationReport:
    """Run one project's evaluation suite."""
    entry = get_project(name)
    if name == "baselines":
        cases = entry.build_cases()
        if split is not None:
            cases = [case for case in cases if case.split is split]
        runner = entry._module().build_runner(provider, kind=baseline_kind)
        report = run_evaluation(
            name, cases, runner, method=baseline_kind,
            model_id=getattr(provider, "model_id", None),
        )
    else:
        report = entry.run(provider, split=split)
    if method:
        report.method = method
    return report


def project_status() -> list[dict[str, Any]]:
    """Status of every project, for ``airl projects``."""
    out: list[dict[str, Any]] = []
    for name in PROJECT_ORDER:
        entry = registry()[name]
        cases = entry.build_cases()
        out.append(
            {
                "name": entry.name,
                "title": entry.title,
                "status": entry.status,
                "description": entry.description,
                "case_count": len(cases),
                "method": entry.method,
            }
        )
    return out
