# naive-judge-baselines

Stripped-down LLM judges, used to test whether the careful prompts earn their keep.

## Why this exists

A careful prompt can flatter a method. The tool prompts are narrow, they name four outcomes, and they are wrapped in abstention rules that downgrade an unjustified contradiction. That is a lot of machinery, and machinery should be measured.

These baselines are the stripped alternatives: the same model, the same cases, the same evidence, with the guidance removed. One is the most common way an LLM grader is wired up, a yes or no question. The other is the three way version, still without any guidance about when to accuse.

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
naive-judge-baselines demo
```

The demo runs the project on one of its own evaluation cases and
prints the input, the verdict, and the reasoning. It uses the
deterministic path, so it needs no model and no network access.

To run the project's own evaluation cases:

```bash
naive-judge-baselines eval
```

## How it works

1. **Context building** gives each baseline exactly the evidence the tool saw, so the comparison is fair.
2. **Prompting** sends the stripped question to the same model.
3. **Scoring** counts the same outcomes, including the cases where a baseline accused where it should have abstained.

## The rules that keep it honest

These are the rules that stop the tool producing false
accusations, which is the failure mode that matters most for a
reliability tool.

* A baseline that abstains where the tool abstains is not a failure. A baseline that accuses where the tool abstains is a false accusation and is counted as one.
* The baselines are not tuned. Nothing here was changed to make a baseline look worse.
* A binary grader has no way to express that the passage does not answer the question, so a no is mapped to `contradicted`. That is the point of the baseline.

## What the results mean

No run file ships with this package. Run the `eval` command to produce one, and read the caveats below before quoting any number it produces.

Every label in this package is model drafted and on the development
split. The numbers are agreement counts over a small tuned case
set, not accuracy estimates, and no independent human review has
been done. Read them as a check that the code does what it says,
not as a claim about how well it works in the world.

## Known limitations

* The comparison is against one model on one case set, so it says something about that configuration, not about models in general.
* The baselines reuse the claim and citation cases, so the comparison covers those two projects.

## Layout

```
naive-judge-baselines/
  README.md
  CHANGELOG.md
  pyproject.toml
  src/baselines/
    baselines.py    the project itself
    cli.py           the demo and eval commands
    demo.py          the worked example
    core/            the shared types, retrieval, and providers
  fixtures/baselines/   the evaluation cases
  tests/            the project's own test suite
```

## Licence

MIT. All fixtures are synthetic.
