"""Deterministic local embedding providers.

Why this exists
---------------
The injected embedding path in ``core/retrieval.py`` had no measured
result because the local Ollama server reports that it does not
support embeddings. That left a hole in the evaluation: the claim
"lexical and embedding retrieval can be compared" was untested.

A network embedding model is not needed to close that hole. A
*hashed* embedding is a real vector space: each text is projected onto
a fixed number of buckets by hashing its tokens, and two texts that
share vocabulary land near each other. It is deterministic, needs no
download, and it is a fair stand-in for testing the *plumbing* of the
embedding path: the ranking code, the cosine similarity, and the
comparison against the lexical ranker.

What it is not
--------------
* It is not a semantic model. "programme" and "course" are different
  tokens and land in different buckets.
* It is not a claim that hashed embeddings are competitive with a
  trained model. The measured result says only that the embedding path
  runs, and where it differs from the lexical path.

A real embedding provider is injected through the same
:class:`~ai_reliability.core.retrieval.EmbeddingProvider` protocol, so
swapping in a hosted model changes nothing but the vectors.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

from .retrieval import tokenize

#: Buckets in the hashed vector space. A larger space reduces collisions
#: at the cost of a sparser vector.
DEFAULT_DIMENSIONS = 512

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")


def _bucket(token: str, dimensions: int) -> tuple[int, float]:
    """Map a token to a bucket and a sign.

    The sign keeps unrelated tokens from all pushing in the same
    direction, which would make every similarity positive and useless
    for ranking.
    """
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % dimensions
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return index, sign


def hashed_embedding(text: str, dimensions: int = DEFAULT_DIMENSIONS) -> list[float]:
    """Return a deterministic hashed vector for one text."""
    vector = [0.0] * dimensions
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]
    if not tokens:
        return vector
    for token in tokens:
        index, sign = _bucket(token, dimensions)
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


class HashedEmbeddingProvider:
    """A deterministic embedding provider with no external dependency.

    It implements the ``EmbeddingProvider`` protocol, so it can be
    passed to :func:`~ai_reliability.core.retrieval.retrieve_semantic`
    or to :func:`~ai_reliability.projects.claims.audit_claims` in
    exactly the same way a hosted embedding model would be.
    """

    model_id = "hashed-embedding-v1"

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS):
        self.dimensions = dimensions
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [hashed_embedding(text, self.dimensions) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        return hashed_embedding(text, self.dimensions)

    @staticmethod
    def tokens(text: str) -> list[str]:
        """The tokenisation this provider uses, exposed for inspection."""
        return tokenize(text)


def cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity of two vectors of equal length."""
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def rank_passages(
    query: str,
    passages: Iterable[str],
    provider: HashedEmbeddingProvider | None = None,
) -> list[tuple[str, float]]:
    """Rank passages by cosine similarity to the query.

    Returns (passage, score) pairs, most similar first. The score is a
    similarity, never evidence of entailment.
    """
    embedder = provider or HashedEmbeddingProvider()
    materialised = list(passages)
    if not materialised:
        return []
    vectors = embedder.embed([query, *materialised])
    query_vector = vectors[0]
    scored = [
        (passage, cosine(query_vector, vector))
        for passage, vector in zip(materialised, vectors[1:])
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored
