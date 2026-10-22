# claim-auditor

Check whether the claims in an AI answer are supported by the sources it was given.

## Why this exists

An AI answer can mix supported, contradicted, and unsupported
statements in the same sentence. A source can discuss exactly the
right topic and still not answer the question. A tool that reports
a contradiction in those cases is worse than no tool, because a
false accusation is a claim the user cannot easily dismiss.

So the tool is built around one rule: a missing passage is never a
contradiction. When the evidence does not settle the question, the
answer is `insufficient_evidence`, and that is reported separately
from `contradicted` so the two can be counted apart.

## Install

Python 3.11 or newer. There are no runtime dependencies.

```bash
pip install .
```

Or run it straight from the source tree:

```bash
export PYTHONPATH=src
```

## Use

```bash
claim-auditor demo
```

The demo runs the project on one of its own evaluation cases and
prints the input, the verdict, and the reasoning. It uses the
deterministic path, so it needs no model and no network access.

To run the project's own evaluation cases:

```bash
claim-auditor eval
```

## How it works

1. **Claim extraction** splits the answer into atomic claims and keeps the qualifiers that change their meaning: dates, quantities, subject, scope, negation, and modality. Offsets into the original answer are preserved exactly.
2. **Evidence retrieval** finds candidate source passages. The retrieval score is topical relevance and nothing more; a passage that shares every keyword can still fail to state the fact the claim needs.
3. **Verdict classification** decides, per claim, whether the evidence supports, contradicts, or does not settle the claim.

## The rules that keep it honest

These are the rules that stop the tool producing false
accusations, which is the failure mode that matters most for a
reliability tool.

* A verdict may only cite evidence that was actually retrieved.
* A universal claim such as all, every, or only is never contradicted by a passage about one example, and never supported by one either.
* A verdict with an empty evidence set cannot be `contradicted`.
* A passage about a different subject cannot carry a verdict, even when the model asserts that two differently named things are the same thing.

## What the results mean

| Metric | Value |
|--------|-------|
| Cases | 11 |
| Scored | 11 |
| Accepted by the fixture scorer | 11 |
| Splits | dev 11 |
| Label origins | model_drafted 11 |
| accepted_outcome | 11 of 11 |
| false_accusations_on_supported | 0 of 1 |
| false_alarms_on_benign | 0 of 0 |
| abstentions | 5 of 11 |

Every label in this package is model drafted and on the development
split. The numbers are agreement counts over a small tuned case
set, not accuracy estimates, and no independent human review has
been done. Read them as a check that the code does what it says,
not as a claim about how well it works in the world.

## Known limitations

* The deterministic path cannot recognise a contradiction. It reports `supported` when a passage contains all of the claim's terms and abstains otherwise. A model is needed for the contradiction cases.
* The claim splitter works on English sentences. It is not a multilingual claim segmenter.
* The tool judges only against the sources it is given. It does not check whether those sources are current or correct.

## Layout

```
claim-auditor/
  README.md
  CHANGELOG.md
  pyproject.toml
  src/claims/
    claims.py    the project itself
    cli.py           the demo and eval commands
    demo.py          the worked example
    core/            the shared types, retrieval, and providers
  fixtures/claims/   the evaluation cases
  tests/            the project's own test suite
```

## Licence

MIT. All fixtures are synthetic.
