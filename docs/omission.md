# Omission auditor

Find what a summary left out that the reader's stated brief cared about.

**Question it answers:** What did this summary leave out that my brief cared about?

**Released as:** `omission_auditor`

## Why it exists

A summary can contain only true statements and still be wrong, because what it left out may be the thing the reader needed. The problem is that an omission has no positive evidence: there is no sentence in the summary to point at.

The tool needs a stated purpose to work from. What was left out is meaningless without one, and it is a different answer for every brief. The same summary against the same source produces different findings depending on what the summary was for, and that is the point rather than a limitation of the method.

## How it works

1. **Segmentation** splits the source into candidate statements, each with the exact span it came from.
2. **Alignment** matches source statements against the summary.
3. **Relevance ranking** scores the unmatched statements against the stated purpose, so what is reported is what the brief cared about, not everything that was left out.

## The rules that keep it honest

* The purpose is required. Without one the tool reports nothing rather than guessing what mattered.
* A finding carries the exact source text and its location, so a reader can check it.
* Benign omissions are not findings. A summary that leaves out irrelevant detail is not an error, and a detector that flags every dropped sentence is useless.

## Known limitations

* Importance is judged against a stated purpose. A summary with no stated purpose cannot be audited for omissions.
* The tool finds omissions in the *source*. It does not know what the reader already knew.
* Two omission cases in the evaluation accept either outcome, because whether a governance duty counts as a security duty is a judgement the tool does not make for you.

## Running it

```bash
PYTHONPATH=src python -m ai_reliability.cli audit eval omission --offline
```

To build the standalone package:

```bash
PYTHONPATH=src python -m ai_reliability.cli release omission --out releases
```
