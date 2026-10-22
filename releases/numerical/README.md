# numerical-verifier

Check that a number in an answer came from the right row, column, year, and unit.

## Why this exists

A number in an answer is the most likely thing to be wrong and the hardest for a reader to check. The value is right, but it came from last year's column, or the wrong company, or the wrong metric, or it is off by a factor of a thousand.

The verifier identifies the row, the column, and the unit from the claim itself, and it shows its arithmetic. A number that cannot be tied to a single cell is not asserted, because picking the nearest match is how a wrong number gets in.

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
numerical-verifier demo
```

The demo runs the project on one of its own evaluation cases and
prints the input, the verdict, and the reasoning. It uses the
deterministic path, so it needs no model and no network access.

To run the project's own evaluation cases:

```bash
numerical-verifier eval
```

## How it works

1. **Table parsing** reads the table and labels its rows and columns.
2. **Claim parsing** identifies the entity, the metric, the period, the value, and the unit from the claim text.
3. **Verification** selects the cell, compares the values, and shows the arithmetic for derived claims such as a percentage change.

## The rules that keep it honest

These are the rules that stop the tool producing false
accusations, which is the failure mode that matters most for a
reliability tool.

* A claim that does not identify a row or a year is reported as uncertain, not matched to the nearest cell.
* A derived claim must show the arithmetic that produced it.
* A scale word in a claim and no unit in the table is a mismatch the tool abstains on, because the table does not say what the numbers are.

## What the results mean

| Metric | Value |
|--------|-------|
| Cases | 11 |
| Scored | 11 |
| Accepted by the fixture scorer | 11 |
| Splits | dev 11 |
| Label origins | model_drafted 11 |
| accepted_outcome | 11 of 11 |
| false_accusations_on_supported | 0 of 3 |
| false_alarms_on_benign | 0 of 0 |
| abstentions | 2 of 11 |

Every label in this package is model drafted and on the development
split. The numbers are agreement counts over a small tuned case
set, not accuracy estimates, and no independent human review has
been done. Read them as a check that the code does what it says,
not as a claim about how well it works in the world.

## Known limitations

* Only straightforward tables are read. Merged cells, footnotes, and multi-level headers are not interpreted.
* The verifier checks arithmetic and cell selection, not whether the source table is itself true.
* Percent changes are computed from the two stated values. It does not model compounding or fiscal calendars.

## Layout

```
numerical-verifier/
  README.md
  CHANGELOG.md
  pyproject.toml
  src/numerical/
    numerical.py    the project itself
    cli.py           the demo and eval commands
    demo.py          the worked example
    core/            the shared types, retrieval, and providers
  fixtures/numerical/   the evaluation cases
  tests/            the project's own test suite
```

## Licence

MIT. All fixtures are synthetic.
