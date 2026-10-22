# Deliverable checker

Check a generated deliverable against every requirement in the request that produced it.

**Question it answers:** Did the output meet every requirement in the request?

**Released as:** `deliverable_checker`

## Why it exists

A generated deliverable can read fluently and still miss what was asked: a missing answer key, three questions where fifty were wanted, a format that was not the one requested. The fluency is exactly what hides the gap.

The checker turns the request into explicit requirement records and checks them one at a time. The rule that makes it usable is that it never invents a requirement. A checker that failed a correct deliverable for a rule the user never gave would be worse than no checker at all.

## How it works

1. **Requirement extraction** turns the request into records that say what is required and whether it can be checked mechanically or needs judgement.
2. **Structural checking** verifies counts, headings, required fields, and named sections against the deliverable.
3. **Reporting** shows every requirement, what was observed, and which ones still need a human.

## The rules that keep it honest

* Every requirement traces back to something in the request. No requirement is invented.
* A requirement that needs judgement is reported as `cannot_verify`, never guessed at.
* A request with no extractable requirements produces `cannot_verify`, not `passed`. Reporting a pass for a vague request would be a false assurance.
* The extracted list is editable, and the requirement IDs survive a recheck, so a correction can be traced.

## Known limitations

* Requirement extraction from free-form prose is the weakest link. A request that says make it good yields no requirements.
* The checks confirm structure, not quality. Fifty questions of nonsense passes the count check.
* Only English request phrasing is understood.

## Running it

```bash
PYTHONPATH=src python -m ai_reliability.cli audit eval deliverables --offline
```

To build the standalone package:

```bash
PYTHONPATH=src python -m ai_reliability.cli release deliverables --out releases
```
