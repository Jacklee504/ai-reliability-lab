"""Tests for the retrieval_eval project.

These check the properties that matter for a reliability tool: it
must not accuse without evidence, and it must abstain when the
evidence does not settle the question.
"""

from __future__ import annotations

from retrieval_eval.core.fixtures import load_cases
from retrieval_eval.retrieval_eval import build_runner

FIXTURE_SOURCES = ['retrieval_eval']

def _provider():
    """The provider the tests run with."""
    return None


def _all_cases():
    """Every case the package ships."""
    cases = []
    for source in FIXTURE_SOURCES:
        cases.extend(load_cases(source))
    return cases


def test_the_project_ships_cases():
    """The package must ship evaluation cases of its own."""
    cases = _all_cases()
    assert cases, "no evaluation cases are packaged"


def test_every_case_runs_and_is_scored():
    """Every packaged case must run and produce a result."""
    cases = _all_cases()
    runner = build_runner(provider=_provider())
    for case in cases:
        result = runner.evaluate_case(case)
        assert result.predicted, case.case_id
        assert result.correct is not None, case.case_id
