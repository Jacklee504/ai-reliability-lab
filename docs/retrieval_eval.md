# Retrieval comparison

Measure whether a ranker retrieves the passage that actually answers the query.

**Question it answers:** Does the ranker retrieve the passage that answers the query?

**Released as:** `retrieval_eval`

## Why it exists

A tool can retrieve the right passage and still judge it wrongly, and it can judge correctly while having retrieved the wrong passage. The two failures look the same from the outside unless retrieval is measured on its own.

This project isolates the ranking stage. Each case supplies a query, a set of candidate passages, and the span that actually answers it, so the two rankers can be compared on the same question: did the right passage make the cut, and how high did it rank?

## How it works

1. **Candidate selection** takes the labelled candidate set and the span that answers the query.
2. **Ranking** runs each method over the same candidates.
3. **Scoring** reports recall at k and mean reciprocal rank for each method.

## The rules that keep it honest

* A retrieval score is topical relevance, never evidence of entailment.
* The candidate set is supplied, so the measurement isolates ranking and says nothing about chunking or source selection.

## Known limitations

* The embedding provider here is a deterministic hashed embedder, not a trained model. It has no synonymy, so it measures the mechanism rather than semantic retrieval.
* The candidate set is given, so the measurement does not cover the whole retrieval pipeline.

## Running it

```bash
PYTHONPATH=src python -m ai_reliability.cli audit eval retrieval_eval --offline
```

To build the standalone package:

```bash
PYTHONPATH=src python -m ai_reliability.cli release retrieval_eval --out releases
```
