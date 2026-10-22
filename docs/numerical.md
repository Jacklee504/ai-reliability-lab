# Numerical verifier

Check that a number in an answer came from the right row, column, year, and unit.

**Question it answers:** Did the number come from the right row, column, year, and unit?

**Released as:** `numerical_verifier`

## Why it exists

A number in an answer is the most likely thing to be wrong and the hardest for a reader to check. The value is right, but it came from last year's column, or the wrong company, or the wrong metric, or it is off by a factor of a thousand.

The verifier identifies the row, the column, and the unit from the claim itself, and it shows its arithmetic. A number that cannot be tied to a single cell is not asserted, because picking the nearest match is how a wrong number gets in.

## How it works

1. **Table parsing** reads the table and labels its rows and columns.
2. **Claim parsing** identifies the entity, the metric, the period, the value, and the unit from the claim text.
3. **Verification** selects the cell, compares the values, and shows the arithmetic for derived claims such as a percentage change.

## The rules that keep it honest

* A claim that does not identify a row or a year is reported as uncertain, not matched to the nearest cell.
* A derived claim must show the arithmetic that produced it.
* A scale word in a claim and no unit in the table is a mismatch the tool abstains on, because the table does not say what the numbers are.

## Known limitations

* Only straightforward tables are read. Merged cells, footnotes, and multi-level headers are not interpreted.
* The verifier checks arithmetic and cell selection, not whether the source table is itself true.
* Percent changes are computed from the two stated values. It does not model compounding or fiscal calendars.

## Running it

```bash
PYTHONPATH=src python -m ai_reliability.cli audit eval numerical --offline
```

To build the standalone package:

```bash
PYTHONPATH=src python -m ai_reliability.cli release numerical --out releases
```
