"""Naive LLM-judge baselines.

Why this exists
---------------
The tool prompts are careful. They name four outcomes, they tell the model
when not to accuse, and they are wrapped in abstention rules that downgrade
an unjustified contradiction. A careful prompt can flatter a method.

This module builds the *stripped* alternatives: the same model, the same
cases, the same evidence, but a prompt with none of the guidance. If the
naive judge scores the same as the careful tool, the tool is not earning
its complexity and the honest thing is to say so.

Two baselines:

``naive_binary``
    "Is this claim supported by the passage? Answer yes or no." The most
    common way an LLM grader is wired up.

``naive_verdict``
    "Answer supported, contradicted, or not enough information." The
    three-way version, still without any guidance about when to accuse.

Deliberate limits
-----------------
* These baselines are the point of the exercise, so they are not tuned.
  Nothing here was changed to make a baseline look worse.
* A binary grader has no way to express "the passage does not answer
  this", so a "no" is mapped to *contradicted*. That is the point of the
  baseline: it must either accuse or refuse to answer.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from ..core.providers import (
    ModelResult,
    Provider,
    ProviderError,
)
from ..core.types import CaseResult, EvalCase, Verdict

BASELINE_PROMPT_VERSION = "naive-baseline-v1"


def build_naive_binary_prompt(claim: str, passages: list[str]) -> str:
    """The stripped prompt: one question, a yes or no answer."""
    evidence = "\n\n".join(passages) if passages else "(none)"
    return (
        "Claim: " + claim + "\n\n"
        "Passage:\n" + evidence + "\n\n"
        "Is this claim supported by the passage? Answer yes or no."
    )


def build_naive_verdict_prompt(claim: str, passages: list[str]) -> str:
    """The stripped three-way prompt."""
    evidence = "\n\n".join(passages) if passages else "(none)"
    return (
        "Claim: " + claim + "\n\n"
        "Passage:\n" + evidence + "\n\n"
        "Is the claim supported, contradicted, or is there not enough "
        "information? Answer with one of those three words."
    )


def _parse_binary(text: str) -> Verdict | None:
    """Map a yes/no answer onto the four-outcome scale."""
    lowered = text.strip().lower()
    first = lowered.split("\n")[0].strip()
    if re.match(r"^yes\b", first):
        return Verdict.SUPPORTED
    if re.match(r"^no\b", first):
        return Verdict.CONTRADICTED
    return None


def _parse_three_way(text: str) -> Verdict | None:
    """Map a three-word answer onto the four-outcome scale."""
    lowered = text.strip().lower()
    first = lowered.split("\n")[0].strip()
    if "not enough" in first or "insufficient" in first:
        return Verdict.INSUFFICIENT
    if "not supported" in first:
        return Verdict.INSUFFICIENT
    if "contradict" in first:
        return Verdict.CONTRADICTED
    if "unsupported" in first:
        return None
    if "support" in first:
        return Verdict.SUPPORTED
    return None


@dataclass
class BaselineResult:
    """The outcome of one baseline call."""

    verdict: Verdict | None
    raw: str = ""
    error: str | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value if self.verdict else None,
            "raw": self.raw,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
        }

def run_naive_baseline(
    provider: Provider,
    claim_text: str,
    passages: list[str],
    kind: str = "naive_binary",
) -> BaselineResult:
    """Run one baseline call and parse its answer.

    ``kind`` is "naive_binary" or "naive_verdict". Compare the resulting
    case IDs with the claim and citation project runs.
    """
    if kind == "naive_binary":
        prompt = build_naive_binary_prompt(claim_text, passages)
        parser = _parse_binary
    elif kind == "naive_verdict":
        prompt = build_naive_verdict_prompt(claim_text, passages)
        parser = _parse_three_way
    else:
        return BaselineResult(verdict=None, error=f"unknown baseline kind {kind!r}")

    try:
        result: ModelResult = provider.generate(
            prompt,
            purpose=kind,
            prompt_version=BASELINE_PROMPT_VERSION,
            temperature=0.0,
            json_mode=(kind == "careful"),
        )
    except ProviderError as exc:
        return BaselineResult(verdict=None, error=f"provider error: {exc}")

    verdict = parser(result.text)
    if verdict is None:
        return BaselineResult(
            verdict=None,
            raw=result.text,
            error="could not parse the answer",
            latency_ms=result.call.latency_ms,
        )
    return BaselineResult(verdict=verdict, raw=result.text, latency_ms=result.call.latency_ms)


def _claim_contexts(inputs: dict[str, Any]) -> list[tuple[Any, list[str]]]:
    """Build the claim list and the passages for each claim.

    This mirrors what the claim auditor does, so a baseline sees exactly
    the same evidence the tool saw.
    """
    from ..core.claims import ExtractionSettings, split_answer_into_claims
    from ..core.retrieval import retrieve
    from .claims import SourceDocument

    answer = inputs.get("answer", "")
    documents = [
        SourceDocument(
            identifier=item.get("identifier", "source"),
            text=item.get("text", ""),
        )
        for item in inputs.get("sources", [])
    ]
    all_spans: list[Any] = []
    for document in documents:
        all_spans.extend(document.spans())

    claims = split_answer_into_claims(answer, settings=ExtractionSettings())
    contexts: list[tuple[Any, list[str]]] = []
    for claim in claims:
        evidence = retrieve(claim.text, all_spans, top_k=4)
        passages = [span.text for ev in evidence for span in ev.spans]
        contexts.append((claim, passages))
    return contexts


@dataclass
class BaselineRunner:
    """Runs the naive baselines over the claim and citation cases.

    The baseline is scored the same way the tool is scored, on the same
    cases, so the numbers are directly comparable. A baseline that
    abstains where the tool abstains is not a failure; a baseline that
    accuses where the tool abstains is a false accusation and is counted
    as one.
    """

    provider: Any = None
    kind: str = "naive_binary"

    def evaluate_case(self, case: EvalCase) -> CaseResult:
        inputs = case.input
        if "citations" in inputs:
            return self._evaluate_citations(case, inputs)
        return self._evaluate_claims(case, inputs)

    def _evaluate_claims(self, case: EvalCase, inputs: dict[str, Any]) -> CaseResult:
        contexts = _claim_contexts(inputs)
        if not contexts:
            return CaseResult(
                case_id=case.case_id,
                project=case.project,
                predicted={"verdict": "no_claims"},
                expected=case.expected,
                label_origin=case.label_origin,
                split=case.split,
                correct=None,
                notes="the answer produced no claims to check",
            )

        verdicts: list[str] = []
        notes: list[str] = []
        raw_answers: list[str] = []
        latency_ms = 0.0
        calls = 0
        failures = 0

        for claim, passages in contexts:
            outcome = run_naive_baseline(
                self.provider, claim.text, passages, kind=self.kind
            )
            calls += 1
            raw_answers.append(outcome.raw)
            latency_ms += outcome.latency_ms
            if outcome.verdict is None:
                failures += 1
                verdicts.append("unparseable")
                notes.append(f"{claim.text!r}: {outcome.error}")
                continue
            verdicts.append(outcome.verdict.value)
            notes.append(f"{claim.text!r}: {outcome.verdict.value}")

        predicted = {
            "verdict": verdicts[0] if len(verdicts) == 1 else "multiple",
            "verdicts": verdicts,
            "raw_answers": raw_answers,
        }
        return self._score(case, predicted, calls, failures, notes, latency_ms)

    def _evaluate_citations(self, case: EvalCase, inputs: dict[str, Any]) -> CaseResult:
        verdicts: list[str] = []
        notes: list[str] = []
        raw_answers: list[str] = []
        latency_ms = 0.0
        calls = 0
        failures = 0

        for item in inputs.get("citations", []):
            outcome = run_naive_baseline(
                self.provider,
                item.get("claim", ""),
                list(item.get("passages", [])),
                kind=self.kind,
            )
            calls += 1
            raw_answers.append(outcome.raw)
            latency_ms += outcome.latency_ms
            if outcome.verdict is None:
                failures += 1
                verdicts.append("unparseable")
                notes.append(f"{item.get('claim', '')!r}: {outcome.error}")
                continue
            verdicts.append(outcome.verdict.value)
            notes.append(f"{item.get('claim', '')!r}: {outcome.verdict.value}")

        predicted = {
            "verdict": verdicts[0] if len(verdicts) == 1 else "multiple",
            "verdicts": verdicts,
            "raw_answers": raw_answers,
        }
        return self._score(case, predicted, calls, failures, notes, latency_ms)

    def _score(
        self,
        case: EvalCase,
        predicted: dict[str, Any],
        calls: int,
        failures: int,
        notes: list[str],
        latency_ms: float,
    ) -> CaseResult:
        """Score a baseline case the same way the tool cases are scored."""
        expected = case.expected
        wanted = expected.get("verdict")
        got = predicted.get("verdict")

        if "verdicts" in expected:
            correct = predicted.get("verdicts") == expected["verdicts"]
        elif wanted is None:
            correct = None
        elif isinstance(wanted, list):
            correct = bool(predicted.get("verdicts")) and all(
                verdict in wanted for verdict in predicted["verdicts"]
            )
        else:
            correct = len(predicted.get("verdicts", [])) == 1 and got == wanted

        return CaseResult(
            case_id=case.case_id,
            project=case.project,
            predicted=predicted,
            expected=expected,
            label_origin=case.label_origin,
            split=case.split,
            correct=correct,
            model_calls=calls,
            latency_ms=latency_ms,
            parse_failures=failures,
            notes="\n".join(notes),
        )


def build_cases() -> list[EvalCase]:
    """The baseline uses the same cases as the tools it compares against."""
    from ..core.fixtures import load_project_cases

    cases: list[EvalCase] = []
    for project in ("claims", "citations"):
        cases.extend(load_project_cases(project))
    return cases


def build_runner(provider: Any = None, kind: str = "naive_binary") -> BaselineRunner:
    """Factory used by the project registry."""
    return BaselineRunner(provider=provider, kind=kind)


def build_prompts(fixture_path: str) -> list[tuple[str, str]]:
    """Build the (purpose, prompt) pairs for a baseline fixture.

    The purpose is namespaced by baseline kind so a recorded cache holds
    both naive kinds without them colliding. The careful method already
    has its own prompt builder in the claim and citation projects.
    """
    import json
    from pathlib import Path

    from ..core.fixtures import project_fixture_dir

    target = Path(fixture_path)
    if not target.exists():
        target = project_fixture_dir("claims") / fixture_path
    payload = json.loads(target.read_text(encoding="utf-8"))

    prompts: list[tuple[str, str]] = []
    for case in payload.get("cases", []):
        inputs = case.get("input", {})
        if "citations" in inputs:
            contexts = [
                (item.get("claim", ""), list(item.get("passages", [])))
                for item in inputs["citations"]
            ]
        else:
            contexts = [(claim.text, passages) for claim, passages in _claim_contexts(inputs)]
        for claim_text, passages in contexts:
            prompts.append((
                "naive_binary",
                build_naive_binary_prompt(claim_text, passages),
            ))
            prompts.append((
                "naive_verdict",
                build_naive_verdict_prompt(claim_text, passages),
            ))
    return prompts
