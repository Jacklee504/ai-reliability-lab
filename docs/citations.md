# Citation auditor

Check whether a cited passage actually supports the sentence it is attached to.

**Question it answers:** Does this cited passage actually support the sentence it is attached to?

**Released as:** `citation_auditor`

## Why it exists

A citation that looks right and supports nothing is the most common failure in a written answer, and it is invisible to the reader. The passage mentions every keyword and simply does not state the fact.

This tool separates two things that are usually conflated: the citation is *topically relevant*, and the citation *supports the claim*. A passage can be perfectly on topic and still fail, and that is `insufficient_evidence`, not `supported`.

## How it works

1. **Passage alignment** attaches each cited passage to the sentence it is cited for.
2. **Topical scoring** measures how much vocabulary the passage shares with the claim, reported separately from support.
3. **Support checking** decides whether the passage states the claim, contradicts it, or leaves it open. Two passages may together support a claim that neither supports alone.

## The rules that keep it honest

* Topical overlap is never reported as support.
* A citation that is absent is `insufficient_evidence`, not a contradiction. A missing citation is a failure mode, not a truth claim.
* Partial support is not full support.
* A citation about a different entity cannot support the claim.

## Known limitations

* The judge works on the passages given to it. It cannot tell you that a better passage exists elsewhere in the document.
* The deterministic path cannot recognise a contradiction from term overlap.
* The tool does not check that the cited source is real, current, or says what it is claimed to say outside the quoted passage.

## Running it

```bash
PYTHONPATH=src python -m ai_reliability.cli audit eval citations --offline
```

To build the standalone package:

```bash
PYTHONPATH=src python -m ai_reliability.cli release citations --out releases
```
