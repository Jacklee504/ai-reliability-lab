# citation-auditor

Check whether a cited passage actually supports the sentence it is attached to.

## Why this exists

A citation that looks right and supports nothing is the most common failure in a written answer, and it is invisible to the reader. The passage mentions every keyword and simply does not state the fact.

This tool separates two things that are usually conflated: the citation is *topically relevant*, and the citation *supports the claim*. A passage can be perfectly on topic and still fail, and that is `insufficient_evidence`, not `supported`.

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
citation-auditor demo
```

The demo runs the project on one of its own evaluation cases and
prints the input, the verdict, and the reasoning. It uses the
deterministic path, so it needs no model and no network access.

To run the project's own evaluation cases:

```bash
citation-auditor eval
```

## How it works

1. **Passage alignment** attaches each cited passage to the sentence it is cited for.
2. **Topical scoring** measures how much vocabulary the passage shares with the claim, reported separately from support.
3. **Support checking** decides whether the passage states the claim, contradicts it, or leaves it open. Two passages may together support a claim that neither supports alone.

## The rules that keep it honest

These are the rules that stop the tool producing false
accusations, which is the failure mode that matters most for a
reliability tool.

* Topical overlap is never reported as support.
* A citation that is absent is `insufficient_evidence`, not a contradiction. A missing citation is a failure mode, not a truth claim.
* Partial support is not full support.
* A citation about a different entity cannot support the claim.

## What the results mean

| Metric | Value |
|--------|-------|
| Cases | 7 |
| Scored | 7 |
| Accepted by the fixture scorer | 7 |
| Splits | dev 7 |
| Label origins | model_drafted 7 |
| accepted_outcome | 7 of 7 |
| false_accusations_on_supported | 0 of 2 |
| false_alarms_on_benign | 0 of 0 |
| abstentions | 4 of 7 |

Every label in this package is model drafted and on the development
split. The numbers are agreement counts over a small tuned case
set, not accuracy estimates, and no independent human review has
been done. Read them as a check that the code does what it says,
not as a claim about how well it works in the world.

## Known limitations

* The judge works on the passages given to it. It cannot tell you that a better passage exists elsewhere in the document.
* The deterministic path cannot recognise a contradiction from term overlap.
* The tool does not check that the cited source is real, current, or says what it is claimed to say outside the quoted passage.

## Layout

```
citation-auditor/
  README.md
  CHANGELOG.md
  pyproject.toml
  src/citations/
    citations.py    the project itself
    cli.py           the demo and eval commands
    demo.py          the worked example
    core/            the shared types, retrieval, and providers
  fixtures/citations/   the evaluation cases
  tests/            the project's own test suite
```

## Licence

MIT. All fixtures are synthetic.
