# Naive judge baselines

Stripped-down LLM judges, used to test whether the careful prompts earn their keep.

**Question it answers:** Do the careful prompts earn their keep, or would a simple judge do as well?

**Released as:** `naive_judge_baselines`

## Why it exists

A careful prompt can flatter a method. The tool prompts are narrow, they name four outcomes, and they are wrapped in abstention rules that downgrade an unjustified contradiction. That is a lot of machinery, and machinery should be measured.

These baselines are the stripped alternatives: the same model, the same cases, the same evidence, with the guidance removed. One is the most common way an LLM grader is wired up, a yes or no question. The other is the three way version, still without any guidance about when to accuse.

## How it works

1. **Context building** gives each baseline exactly the evidence the tool saw, so the comparison is fair.
2. **Prompting** sends the stripped question to the same model.
3. **Scoring** counts the same outcomes, including the cases where a baseline accused where it should have abstained.

## The rules that keep it honest

* A baseline that abstains where the tool abstains is not a failure. A baseline that accuses where the tool abstains is a false accusation and is counted as one.
* The baselines are not tuned. Nothing here was changed to make a baseline look worse.
* A binary grader has no way to express that the passage does not answer the question, so a no is mapped to `contradicted`. That is the point of the baseline.

## Known limitations

* The comparison is against one model on one case set, so it says something about that configuration, not about models in general.
* The baselines reuse the claim and citation cases, so the comparison covers those two projects.

## Running it

```bash
PYTHONPATH=src python -m ai_reliability.cli audit eval baselines --offline
```

To build the standalone package:

```bash
PYTHONPATH=src python -m ai_reliability.cli release baselines --out releases
```
