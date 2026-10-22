# Claim auditor

Check whether the claims in an AI answer are supported by the sources it was given.

**Question it answers:** Which sentences in this answer are not supported by the source?

**Released as:** `claim_auditor`

## Why it exists

An AI answer can mix supported, contradicted, and unsupported
statements in the same sentence. A source can discuss exactly the
right topic and still not answer the question. A tool that reports
a contradiction in those cases is worse than no tool, because a
false accusation is a claim the user cannot easily dismiss.

So the tool is built around one rule: a missing passage is never a
contradiction. When the evidence does not settle the question, the
answer is `insufficient_evidence`, and that is reported separately
from `contradicted` so the two can be counted apart.

## How it works

1. **Claim extraction** splits the answer into atomic claims and keeps the qualifiers that change their meaning: dates, quantities, subject, scope, negation, and modality. Offsets into the original answer are preserved exactly.
2. **Evidence retrieval** finds candidate source passages. The retrieval score is topical relevance and nothing more; a passage that shares every keyword can still fail to state the fact the claim needs.
3. **Verdict classification** decides, per claim, whether the evidence supports, contradicts, or does not settle the claim.

## The rules that keep it honest

* A verdict may only cite evidence that was actually retrieved.
* A universal claim such as all, every, or only is never contradicted by a passage about one example, and never supported by one either.
* A verdict with an empty evidence set cannot be `contradicted`.
* A passage about a different subject cannot carry a verdict, even when the model asserts that two differently named things are the same thing.

## Known limitations

* The deterministic path cannot recognise a contradiction. It reports `supported` when a passage contains all of the claim's terms and abstains otherwise. A model is needed for the contradiction cases.
* The claim splitter works on English sentences. It is not a multilingual claim segmenter.
* The tool judges only against the sources it is given. It does not check whether those sources are current or correct.

## Running it

```bash
PYTHONPATH=src python -m ai_reliability.cli audit eval claims --offline
```

To build the standalone package:

```bash
PYTHONPATH=src python -m ai_reliability.cli release claims --out releases
```
