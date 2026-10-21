"""Retrieval method comparison: lexical ranking against embedding ranking.

Problem
-------
The lab separates retrieval from judgement so that a wrong answer can
be attributed to the stage that caused it. That separation is only
useful if the retrieval stage is itself measured, and so far it was
not: the embedding path existed but had never been run against
labelled cases, so a claim about it could not be checked.

Approach
--------
A retrieval case supplies a query and a set of candidate passages, plus
the span that actually answers the query. Both rankers select the same
number of passages and are scored on the same question: did the
answering passage make the cut?

Two questions are reported, because they are different:

recall at k
    Did the answering passage appear in the selected set at all? This
    is the question that matters for the claim auditor: a passage
    that is not retrieved can never be judged correctly.

reciprocal rank
    How highly was the answering passage ranked? A high recall with a
    low rank means the right passage was retrieved but buried.

Deliberate limits
-----------------
* The embedding provider is a deterministic hashed embedder, not a
  trained model. It has no synonymy, so it is a test of the
  *mechanism*, not a claim about semantic retrieval. See
  ``core/embeddings.py`` for why.
* A retrieval score is never an entailment score. A method that
  retrieves the right passage can still judge it wrongly, and this
  project never says it succeeded.
* The candidate set is supplied, not built. This measures *ranking*
  within a fixed set, which is the part the two methods can differ
  on. It does not measure chunking or source selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.embeddings import HashedEmbeddingProvider
from ..core.fixtures import load_project_cases
from ..core.retrieval import retrieve, retrieve_semantic
from ..core.types import CaseResult, EvalCase, Span, short_id


def _spans(case: EvalCase) -> list[Span]:
    """Build the candidate spans for a retrieval case."""
    spans: list[Span] = []
    for index, passage in enumerate(case.input.get("passages", []), start=1):
        spans.append(
            Span(
                span_id=short_id("sp", case.case_id, str(index)),
                source_id=case.case_id,
                text=passage,
                start=0,
                end=len(passage),
                kind="candidate",
            )
        )
    return spans


def _answer_indices(case: EvalCase) -> set[int]:
    """Which candidate indices answer the query.

    A case may list several answering passages, for example a query
    that needs two passages together.
    """
    raw = case.input.get("answering_passages", [])
    if isinstance(raw, int):
        raw = [raw]
    return {int(index) for index in raw}


@dataclass
class RetrievalOutcome:
    """How one ranker did on one case."""

    method: str
    selected: list[int]
    rank: int | None
    hit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "selected": self.selected,
            "rank": self.rank,
            "hit": self.hit,
        }


def evaluate_retrieval(
    case: EvalCase,
    *,
    top_k: int = 3,
    embedder: HashedEmbeddingProvider | None = None,
) -> dict[str, RetrievalOutcome]:
    """Run both rankers over one case."""
    spans = _spans(case)
    query = case.input.get("query", "")
    answers = _answer_indices(case)

    lexical = retrieve(query, spans, top_k=top_k)
    lexical_indices = []
    for evidence in lexical:
        for span in evidence.spans:
            for index, candidate in enumerate(spans, start=1):
                if candidate.span_id == span.span_id:
                    lexical_indices.append(index)
                    break

    semantic = retrieve_semantic(
        query, spans, embedder or HashedEmbeddingProvider(), top_k=top_k
    )
    semantic_indices = []
    for evidence in semantic:
        for span in evidence.spans:
            for index, candidate in enumerate(spans, start=1):
                if candidate.span_id == span.span_id:
                    semantic_indices.append(index)
                    break

    def outcome(method: str, indices: list[int]) -> RetrievalOutcome:
        rank = None
        for position, index in enumerate(indices, start=1):
            if index in answers:
                rank = position
                break
        return RetrievalOutcome(
            method=method,
            selected=indices,
            rank=rank,
            hit=rank is not None,
        )

    return {
        "lexical": outcome("lexical", lexical_indices),
        "embedding": outcome("embedding", semantic_indices),
    }


@dataclass
class RetrievalRunner:
    """Runs the retrieval comparison over a set of cases."""

    provider: Any = None
    top_k: int = 3
    embedder: HashedEmbeddingProvider = field(default_factory=HashedEmbeddingProvider)

    def evaluate_case(self, case: EvalCase) -> CaseResult:
        outcomes = evaluate_retrieval(case, top_k=self.top_k, embedder=self.embedder)
        expected = case.expected

        checks: list[bool] = []
        for method, outcome in outcomes.items():
            if "hit_%s" % method in expected:
                checks.append(outcome.hit == expected["hit_%s" % method])
        correct = all(checks) if checks else None

        predicted = {
            "verdict": "hit" if outcomes["lexical"].hit else "miss",
            "lexical": outcomes["lexical"].to_dict(),
            "embedding": outcomes["embedding"].to_dict(),
        }
        return CaseResult(
            case_id=case.case_id,
            project=case.project,
            predicted=predicted,
            expected=expected,
            label_origin=case.label_origin,
            split=case.split,
            correct=correct,
            notes=(
                "lexical rank=%s selected=%s | embedding rank=%s selected=%s"
                % (
                    outcomes["lexical"].rank,
                    outcomes["lexical"].selected,
                    outcomes["embedding"].rank,
                    outcomes["embedding"].selected,
                )
            ),
        )


def build_cases() -> list[EvalCase]:
    """Load the retrieval comparison cases."""
    return load_project_cases("retrieval_eval")


def build_runner(provider: Any = None) -> RetrievalRunner:
    """Factory used by the project registry."""
    return RetrievalRunner(provider=provider)


def summarise(results: list[CaseResult]) -> dict[str, Any]:
    """Summarise a retrieval run: recall at k and reciprocal rank."""
    out: dict[str, Any] = {"methods": {}}
    for method in ("lexical", "embedding"):
        hits = 0
        reciprocal = 0.0
        total = 0
        for result in results:
            block = result.predicted.get(method)
            if not block:
                continue
            total += 1
            if block.get("hit"):
                hits += 1
                rank = block.get("rank") or 1
                reciprocal += 1.0 / rank
        out["methods"][method] = {
            "cases": total,
            "recall_at_k": hits,
            "recall_denominator": total,
            "mean_reciprocal_rank": round(reciprocal / total, 4) if total else None,
        }
    return out
