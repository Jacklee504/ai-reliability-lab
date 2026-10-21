"""Evidence retrieval.

Retrieval is kept separate from judgement on purpose: the evaluation
report needs to distinguish "found the wrong passage" from "judged the
passage wrongly". Retrieval here is deliberately simple and transparent.

Two methods:

``lexical``
    Token overlap with IDF weighting, plus a bonus for matching numbers
    and dates. Deterministic, free, and the baseline every model-assisted
    method has to beat.

``embedding``
    An injected embedding provider can rank the same spans by cosine
    similarity. No endpoint or embedding model is bundled, and this path
    has no measured result yet.

A retrieval score is *not* evidence of entailment. The verdict stage
always has to look at the passage text.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from .types import Evidence, Span, short_id


class EmbeddingProvider(Protocol):
    """Injectable embedder for retrieval experiments."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


def retrieve_semantic(
    claim_text: str,
    spans: list[Span],
    embedder: EmbeddingProvider,
    *,
    top_k: int = 5,
) -> list[Evidence]:
    """Rank exact source spans by cosine similarity, without judging entailment."""
    if not spans:
        return []
    vectors = embedder.embed([claim_text, *[span.text for span in spans]])
    if len(vectors) != len(spans) + 1 or not vectors[0]:
        raise ValueError("embedder returned the wrong number of vectors")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("embedding dimensions differ")
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise ValueError("embeddings must contain finite numbers")
    query = vectors[0]
    query_norm = math.sqrt(sum(value * value for value in query))
    scored = []
    for span, vector in zip(spans, vectors[1:]):
        norm = math.sqrt(sum(value * value for value in vector))
        similarity = sum(a * b for a, b in zip(query, vector)) / (query_norm * norm) if query_norm and norm else 0.0
        scored.append((similarity, span))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        Evidence(
            evidence_id=short_id("ev", claim_text, span.span_id, f"{score:.6f}"),
            claim_id="", spans=(span,), retrieval_score=round(score, 6),
            method="embedding",
        )
        for score, span in scored[:top_k] if score > 0.0
    ]

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'’\-]*")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these",
    "those", "it", "its", "as", "at", "by", "from", "has", "have", "had", "will",
    "would", "shall", "should", "may", "might", "can", "could", "do", "does",
    "did", "not", "no", "if", "then", "than", "so", "such", "there", "their",
    "they", "them", "he", "she", "we", "you", "i", "all", "any", "each", "more",
    "most", "other", "some", "such", "only", "own", "same", "into", "about",
    "also", "which", "who", "whom", "what", "when", "where", "why", "how",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def lexical_support_candidate(claim_text: str, passage_text: str) -> bool:
    """Require all content tokens and key logical qualifiers before support.

    This is deliberately stricter than topical overlap. It remains only
    a candidate check, so applications should not treat it as entailment.
    """
    claim_tokens = set(tokenize(claim_text))
    passage_tokens = set(tokenize(passage_text))
    if not claim_tokens or not claim_tokens <= passage_tokens:
        return False
    negation = re.compile(r"\b(?:not|never|no|without|cannot)\b", re.IGNORECASE)
    if bool(negation.search(claim_text)) != bool(negation.search(passage_text)):
        return False
    universal = re.compile(r"\b(?:all|every|each|only)\b", re.IGNORECASE)
    if universal.search(claim_text) and not universal.search(passage_text):
        return False
    return not numeric_conflict(claim_text, passage_text)


@dataclass
class IndexedSpan:
    """A candidate passage prepared for retrieval."""

    span: Span
    tokens: Counter
    length: int


def build_index(spans: list[Span]) -> list[IndexedSpan]:
    """Build the IDF-weighted lexical index over candidate spans."""
    token_lists = [tokenize(s.text) for s in spans]
    document_frequency: Counter = Counter()
    for tokens in token_lists:
        for token in set(tokens):
            document_frequency[token] += 1

    total_docs = max(1, len(spans))
    indexed: list[IndexedSpan] = []
    for span, tokens in zip(spans, token_lists):
        counts = Counter(tokens)
        idf = {
            token: math.log((total_docs + 1) / (df + 1)) + 1.0
            for token, df in document_frequency.items()
        }
        weighted = {token: count * idf.get(token, 1.0) for token, count in counts.items()}
        indexed.append(
            IndexedSpan(span=span, tokens=Counter(weighted), length=len(tokens) or 1)
        )
    return indexed


def _number_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d[\d,.]*", text.lower()))
