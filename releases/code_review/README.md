# code-review-verifier

Check whether an AI review comment is technically true of the code it points at.

## Why this exists

An AI review comment can sound precise while being false. It cites a real line number, quotes real code, and still misses a guard a few lines up, a caller elsewhere, or a test that covers the case.

The verifier extracts the checkable assertion from the comment and looks for evidence. It reports what it can establish, and abstains on what it cannot, which is the honest answer for a comment about concurrency or a runtime failure.

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
code-review-verifier demo
```

The demo runs the project on one of its own evaluation cases and
prints the input, the verdict, and the reasoning. It uses the
deterministic path, so it needs no model and no network access.

To run the project's own evaluation cases:

```bash
code-review-verifier eval
```

## How it works

1. **Assertion extraction** separates the checkable claim from the preamble and the advice.
2. **Static checks** look for the guard, the bound, or the unbounded construct the comment says is missing.
3. **Execution** runs the reviewer's own harness in a separate process with a timeout, when the comment is about runtime behaviour.

## The rules that keep it honest

These are the rules that stop the tool producing false
accusations, which is the failure mode that matters most for a
reliability tool.

* A comment about a file that is not in the snapshot is unverified, not wrong and not right.
* A comment that cannot be parsed is unverified. Malformed code never supports an accusation.
* A harness that times out, fails to import, or is absent leaves the claim unverified. A broken harness can never manufacture evidence against a comment.
* Style is not a technical assertion and is never judged.

## What the results mean

| Metric | Value |
|--------|-------|
| Cases | 9 |
| Scored | 9 |
| Accepted by the fixture scorer | 9 |
| Splits | dev 9 |
| Label origins | model_drafted 9 |
| accepted_outcome | 9 of 9 |
| false_accusations_on_supported | 0 of 3 |
| false_alarms_on_benign | 0 of 0 |
| abstentions | 3 of 9 |

Every label in this package is model drafted and on the development
split. The numbers are agreement counts over a small tuned case
set, not accuracy estimates, and no independent human review has
been done. Read them as a check that the code does what it says,
not as a claim about how well it works in the world.

## Known limitations

* The static checks cover a narrow set of structural claims: missing guards, unbounded loops, and the like.
* A concurrency claim cannot be settled by reading a function alone.
* The execution path runs the reviewer's harness, not the project's test suite, so it shows what that harness observed.

## Layout

```
code-review-verifier/
  README.md
  CHANGELOG.md
  pyproject.toml
  src/code_review/
    code_review.py    the project itself
    cli.py           the demo and eval commands
    demo.py          the worked example
    core/            the shared types, retrieval, and providers
  fixtures/code_review/   the evaluation cases
  tests/            the project's own test suite
```

## Licence

MIT. All fixtures are synthetic.
