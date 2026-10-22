# Code review verifier

Check whether an AI review comment is technically true of the code it points at.

**Question it answers:** Is this review comment technically true of the code?

**Released as:** `code_review_verifier`

## Why it exists

An AI review comment can sound precise while being false. It cites a real line number, quotes real code, and still misses a guard a few lines up, a caller elsewhere, or a test that covers the case.

The verifier extracts the checkable assertion from the comment and looks for evidence. It reports what it can establish, and abstains on what it cannot, which is the honest answer for a comment about concurrency or a runtime failure.

## How it works

1. **Assertion extraction** separates the checkable claim from the preamble and the advice.
2. **Static checks** look for the guard, the bound, or the unbounded construct the comment says is missing.
3. **Execution** runs the reviewer's own harness in a separate process with a timeout, when the comment is about runtime behaviour.

## The rules that keep it honest

* A comment about a file that is not in the snapshot is unverified, not wrong and not right.
* A comment that cannot be parsed is unverified. Malformed code never supports an accusation.
* A harness that times out, fails to import, or is absent leaves the claim unverified. A broken harness can never manufacture evidence against a comment.
* Style is not a technical assertion and is never judged.

## Known limitations

* The static checks cover a narrow set of structural claims: missing guards, unbounded loops, and the like.
* A concurrency claim cannot be settled by reading a function alone.
* The execution path runs the reviewer's harness, not the project's test suite, so it shows what that harness observed.

## Running it

```bash
PYTHONPATH=src python -m ai_reliability.cli audit eval code_review --offline
```

To build the standalone package:

```bash
PYTHONPATH=src python -m ai_reliability.cli release code_review --out releases
```
