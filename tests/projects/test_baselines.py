"""Tests for the baselines project.

These challenge behaviour rather than restating the
implementation. The cases that matter are the ones where a
plausible implementation would be wrong: a false accusation, a
false alarm, a qualifier that changes the answer, and a failure
that must not be silently turned into a verdict.
"""

from __future__ import annotations

from ai_reliability.core.providers import FakeProvider
from ai_reliability.core.types import EvalCase, LabelOrigin, Split
from ai_reliability.projects.baselines import BaselineRunner, run_naive_baseline

def test_naive_judges_keep_nonanswers_out_of_binary_verdicts():
    binary = FakeProvider(responses={"naive_binary": "Not enough information."})
    assert run_naive_baseline(binary, "A", ["B"]).verdict is None
    three_way = FakeProvider(responses={"naive_verdict": "Not enough information."})
    assert run_naive_baseline(three_way, "A", ["B"], "naive_verdict").verdict.value == "insufficient_evidence"


def test_multi_claim_baseline_requires_the_expected_sequence():
    case = EvalCase(
        case_id="mixed", project="claims",
        input={"answer": "The fee is 10 EUR and the notice is 30 days.", "sources": []},
        expected={"verdicts": ["supported", "contradicted"]},
    )
    always_yes = FakeProvider(responses={"naive_binary": "Yes"})
    result = BaselineRunner(provider=always_yes).evaluate_case(case)
    assert result.predicted["verdicts"] == ["supported", "supported"]
    assert result.correct is False
