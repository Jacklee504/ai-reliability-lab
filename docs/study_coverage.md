# Study coverage auditor

Check which syllabus topics are explicitly mentioned in a set of study material.

**Question it answers:** Which syllabus topics are explicitly mentioned in the material?

**Released as:** `study_coverage`

## Why it exists

Syllabus coverage is usually checked by eye, and by eye it is easy to accept a description of a topic as coverage of it. A student reading about how a function calls itself has read about recursion, but the word never appears.

The tool proves only the narrow thing it can prove: a topic phrase appeared. It reports a mention or it reports an unverified topic, and it never says a topic was taught well, because a mention is not evidence of understanding.

## How it works

1. **Topic definition** records each syllabus topic with its label and any accepted aliases.
2. **Mention counting** finds the phrase occurrences, respecting word boundaries so a longer word does not count.
3. **Reporting** lists the topics that were mentioned and the ones that were not, without judging either.

## The rules that keep it honest

* A mention is a lexical observation, not a claim about teaching quality.
* A paraphrase is not a mention. The tool reports what it can prove.
* Repetition is counted, but the count is not a measure of emphasis.

## Known limitations

* The tool measures lexical presence, not semantic coverage. A topic described in other words is reported as unverified.
* Aliases are supplied by the caller. A missing alias means a missed mention.
* It says nothing about whether the material teaches the topic well enough.

## Running it

```bash
PYTHONPATH=src python -m ai_reliability.cli audit eval study_coverage --offline
```

To build the standalone package:

```bash
PYTHONPATH=src python -m ai_reliability.cli release study_coverage --out releases
```
