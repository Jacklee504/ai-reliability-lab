"""Project 7: numerical and table-claim verifier.

Problem
-------
An AI can cite a real table and still take a value from the wrong year,
the wrong row, the wrong column, the wrong unit, or the wrong
denominator. The citation is genuine, so a plain similarity check sees a
perfect match and passes it.

Approach
--------
The verifier parses tables into addressable cells first, so a claim is
checked against a *coordinate* (row, column) rather than against a blob
of text. Then it parses the claim into its parts:

* **entity**: which row the claim is about,
* **metric**: which column,
* **period**: which year or quarter,
* **unit**: euros, millions, percent,
* **operation**: a direct read, a difference, or a percentage change.

The verdict names the cells it used and shows the arithmetic for derived
claims, so a reader can check the reasoning rather than trust it.

When the row or column relationship is not unambiguous, the verifier
abstains. It does not guess.

Deliberate limits
-----------------
* Only CSV and Markdown tables are parsed. PDF tables are out of scope
  because column boundaries are not reliably recoverable.
* A claim whose entity, metric, or period cannot be pinned to one cell
  is reported as uncertain, never as supported.
* The verifier checks the *table*, not the world. A correct read of a
  wrong table is still a correct read of that table.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from typing import Any

from ..core.types import CaseResult, CheckStatus, EvalCase, Method, Span, Verdict

#: Multipliers for unit words, used to normalise a cell value.
UNIT_MULTIPLIERS = {
    "": 1.0,
    "unit": 1.0,
    "units": 1.0,
    "thousand": 1_000.0,
    "k": 1_000.0,
    "million": 1_000_000.0,
    "m": 1_000_000.0,
    "million eur": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "bn": 1_000_000_000.0,
    "percent": 1.0,
    "%": 1.0,
}

_NUMBER_RE = re.compile(r"[-+]?\d[\d,.]*")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_PERIOD_RE = re.compile(
    r"\b(Q[1-4]\s*(?:19|20)\d{2}|(?:19|20)\d{2}\s*Q[1-4]|(?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_PERCENT_CHANGE_RE = re.compile(
    r"(?:increased?|decreased?|rose|fell|grew|declined|changed|up|down)\s+by\s+"
    r"([\d,.]+)\s*(?:%|percent)",
    re.IGNORECASE,
)
_DIFFERENCE_RE = re.compile(
    r"difference between|compared with|compared to|more than|less than|higher than|lower than",
    re.IGNORECASE,
)


@dataclass
class Table:
    """A parsed table with addressable cells."""

    identifier: str
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    row_labels: list[str] = field(default_factory=list)

    def cell(self, row: int, column: int) -> str:
        if 0 <= row < len(self.rows) and 0 <= column < len(self.rows[row]):
            return self.rows[row][column]
        return ""

    def find_row(self, label: str, metric: str | None = None) -> int:
        """Find the row for an entity, optionally disambiguated by metric.

        A table with both an entity column and a metric column repeats the
        entity label on several rows, so the metric is needed to pick the
        right one.
        """
        needle = label.strip().lower()
        if metric:
            for index, row in enumerate(self.rows):
                if not row:
                    continue
                if row[0].strip().lower() != needle:
                    continue
                if metric.lower() in row[1].lower():
                    return index
        for index, row_label in enumerate(self.row_labels):
            if row_label.strip().lower() == needle:
                return index
        for index, row_label in enumerate(self.row_labels):
            if needle in row_label.lower():
                return index
        return -1

    def find_column(self, header: str) -> int:
        needle = header.strip().lower()
        for index, name in enumerate(self.headers):
            if needle in name.lower():
                return index
        return -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "headers": self.headers,
            "row_labels": self.row_labels,
            "row_count": len(self.rows),
        }


@dataclass
class NumericClaim:
    """A parsed numeric claim."""

    text: str
    entity: str | None = None
    metric: str | None = None
    period: str | None = None
    unit: str | None = None
    value: float | None = None
    operation: str = "read"  # "read" | "difference" | "percent_change"
    operation_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "entity": self.entity,
            "metric": self.metric,
            "period": self.period,
            "unit": self.unit,
            "value": self.value,
            "operation": self.operation,
            "operation_value": self.operation_value,
        }


@dataclass
class NumericFinding:
    """The verification result for one numeric claim."""

    claim: NumericClaim
    verdict: Verdict
    explanation: str = ""
    cells_used: list[dict[str, Any]] = field(default_factory=list)
    arithmetic: str = ""
    method: Method = Method.DETERMINISTIC

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "verdict": self.verdict.value,
            "explanation": self.explanation,
            "cells_used": self.cells_used,
            "arithmetic": self.arithmetic,
            "method": self.method.value,
        }


def parse_csv_table(text: str, identifier: str = "table") -> Table:
    """Parse a CSV table into headers and rows."""
    reader = csv.reader(io.StringIO(text))
    records = [r for r in reader if any(cell.strip() for cell in r)]
    if not records:
        return Table(identifier=identifier)
    headers = [h.strip() for h in records[0]]
    rows = []
    row_labels = []
    for record in records[1:]:
        padded = list(record) + [""] * (len(headers) - len(record))
        rows.append([c.strip() for c in padded[: len(headers)]])
        row_labels.append(padded[0].strip() if padded else "")
    return Table(identifier=identifier, headers=headers, rows=rows, row_labels=row_labels)


def parse_markdown_table(text: str, identifier: str = "table") -> Table:
    """Parse a Markdown pipe table into headers and rows."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("|")]
    if not lines:
        return Table(identifier=identifier)

    def cells(line: str) -> list[str]:
        inner = line.strip().strip("|")
        return [c.strip() for c in inner.split("|")]

    header_cells = cells(lines[0])
    body = lines[1:]
    if body and all(set(c) <= set("-: ") and c for c in cells(body[0])):
        body = body[1:]
    rows = [cells(line) for line in body]
    row_labels = [r[0] if r else "" for r in rows]
    return Table(
        identifier=identifier,
        headers=header_cells,
        rows=rows,
        row_labels=row_labels,
    )


def parse_table(text: str, identifier: str = "table", fmt: str = "auto") -> Table:
    """Parse a table, choosing the format automatically when possible."""
    if fmt == "csv":
        return parse_csv_table(text, identifier)
    if fmt == "markdown":
        return parse_markdown_table(text, identifier)
    if "|" in text and text.count("|") >= 2:
        return parse_markdown_table(text, identifier)
    return parse_csv_table(text, identifier)


def _parse_value(text: str) -> tuple[float | None, str]:
    """Parse a cell value into a number and a unit.

    Returns (value, unit). A cell that is not numeric yields (None, text).
    """
    cleaned = text.strip()
    if not cleaned:
        return None, ""
    match = _NUMBER_RE.search(cleaned)
    if not match:
        return None, cleaned.lower()
    raw = match.group(0).replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None, cleaned.lower()
    unit = cleaned[match.end():].strip().lower()
    return value, unit


def _extract_claimed_value(text: str, period: str | None) -> float | None:
    """Extract the number the claim asserts, ignoring the period.

    A claim that names a period puts the year before the value, so a
    naive "first number" rule reads the year as the value. The value is
    taken from after the copula when there is one, and otherwise from the
    last number that is not part of the period.
    """
    candidates: list[tuple[int, float]] = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group(0).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if period:
            # Skip a number that is part of the period token.
            start = match.start()
            if start < len(text) and text[start : start + len(period)] == period:
                continue
            if period in text[max(0, start - 2) : start + len(period) + 2]:
                continue
        candidates.append((match.start(), value))

    if not candidates:
        return None

    copula = re.search(r"\b(?:was|were|is|are|equals?|totaled?|reached?|stood at)\b", text, re.IGNORECASE)
    if copula:
        after = [value for start, value in candidates if start > copula.end()]
        if after:
            return after[0]
    return candidates[-1][1]


def _normalise(value: float, unit: str) -> float:
    """Scale a value by its unit multiplier."""
    return value * UNIT_MULTIPLIERS.get(unit.strip().lower(), 1.0)


def parse_numeric_claim(text: str, table: Table) -> NumericClaim:
    """Parse a numeric claim into its parts.

    The entity and metric are matched against the table's row labels and
    headers. When they cannot be pinned down, the claim stays ambiguous,
    which is what makes the verifier abstain rather than guess.
    """
    claim = NumericClaim(text=text)

    period_match = _PERIOD_RE.search(text)
    if period_match:
        claim.period = period_match.group(0).strip()

    percent_change = _PERCENT_CHANGE_RE.search(text)
    if percent_change:
        claim.operation = "percent_change"
        try:
            claim.operation_value = float(percent_change.group(1).replace(",", ""))
        except ValueError:
            claim.operation_value = None
    elif _DIFFERENCE_RE.search(text):
        claim.operation = "difference"

    # The value is the number that the claim asserts, not the first
    # number in the sentence. A sentence like "revenue in 2024 was
    # 5100000" mentions the year first, so the year must be excluded.
    claim.value = _extract_claimed_value(text, claim.period)

    lowered = text.lower()
    for unit in ("million", "thousand", "billion", "percent", "%", "eur", "usd", "gbp"):
        if unit in lowered:
            claim.unit = unit
            break

    # The entity is the row label that the claim mentions, if any.
    for label in table.row_labels:
        if label and label.lower() in lowered:
            claim.entity = label
            break

    # A table shaped like "Entity,Metric,2023,2024" puts the metric
    # value in column 1 rather than in a header, so the metric is taken
    # from the column header only when the table has no metric column.
    has_metric_column = any(
        h.strip().lower() in {"metric", "measure", "kpi", "value"} for h in table.headers
    )

    # The metric is the header that the claim mentions, if any. A
    # header that names a period is not a metric, so it is skipped.
    for header in table.headers[1:]:
        if not header or header.lower() not in lowered:
            continue
        if _PERIOD_RE.search(header) or _YEAR_RE.search(header):
            continue
        claim.metric = header
        break

    # Some tables put the metric in the row and the entity in the column.
    # When the claim mentions a header, and that header is also a row
    # label, the row is the metric row and the header is the entity.
    if claim.metric is None:
        for header in table.headers:
            if header and header.lower() in lowered and header in table.row_labels:
                claim.metric = header
                break

    if claim.metric is None and has_metric_column:
        for row in table.rows:
            if len(row) > 1 and row[1].strip().lower() in lowered:
                claim.metric = row[1].strip()
                break

    return claim


def verify_numeric_claim(
    claim_text: str,
    table: Table,
) -> NumericFinding:
    """Check one numeric claim against one table.

    The verdict names the cells it used and shows the arithmetic, so the
    result can be inspected rather than trusted.
    """
    claim = parse_numeric_claim(claim_text, table)

    if not table.rows or not table.headers:
        return NumericFinding(
            claim=claim,
            verdict=Verdict.INSUFFICIENT,
            explanation="the table is empty or could not be parsed",
        )

    metric_in_column = _has_metric_column(table)
    column_index = -1
    if claim.metric and not metric_in_column:
        column_index = table.find_column(claim.metric)
    row_index = table.find_row(claim.entity, claim.metric) if claim.entity else -1

    if claim.entity and row_index < 0:
        return NumericFinding(
            claim=claim,
            verdict=Verdict.UNCERTAIN,
            explanation=f"the row for {claim.entity!r} is not in the table",
        )
    if claim.metric and column_index < 0 and not metric_in_column:
        return NumericFinding(
            claim=claim,
            verdict=Verdict.UNCERTAIN,
            explanation=f"the column for {claim.metric!r} is not in the table",
        )

    # The period selects the column when the table has period headers.
    # A table shaped like "Entity,Metric,2023,2024" has two leading
    # non-period columns, which the search skips.
    period_column = -1
    if claim.period:
        leading = 2 if _has_metric_column(table) else 1
        for index, header in enumerate(table.headers):
            if index < leading:
                continue
            if claim.period.lower() in header.lower():
                period_column = index
                break

    if claim.operation == "read":
        if row_index < 0 and column_index < 0 and period_column < 0:
            return NumericFinding(
                claim=claim,
                verdict=Verdict.UNCERTAIN,
                explanation=(
                    "the claim does not identify a row, a column, or a period, "
                    "so no cell can be selected"
                ),
            )
        if period_column >= 0:
            column_index = period_column
        if row_index < 0:
            return NumericFinding(
                claim=claim,
                verdict=Verdict.UNCERTAIN,
                explanation="the row of the claim is not determined",
            )
        if column_index < 0:
            return NumericFinding(
                claim=claim,
                verdict=Verdict.UNCERTAIN,
                explanation="the column of the claim is not determined",
            )

        raw = table.cell(row_index, column_index)
        value, unit = _parse_value(raw)
        if value is None:
            return NumericFinding(
                claim=claim,
                verdict=Verdict.UNCERTAIN,
                explanation=f"the cell at row {row_index} column {column_index} is not numeric",
            )

        cell = {
            "row_label": table.row_labels[row_index],
            "column_header": table.headers[column_index],
            "raw": raw,
            "value": value,
            "unit": unit,
        }
        expected = _normalise(value, unit)

        if claim.value is None:
            return NumericFinding(
                claim=claim,
                verdict=Verdict.UNCERTAIN,
                explanation="the claim does not state a number to compare",
                cells_used=[cell],
            )

        scale_words = {"thousand", "k", "million", "m", "billion", "bn"}
        claim_scale = (claim.unit or "").strip().lower() in scale_words
        cell_scale = (unit or "").strip().lower() in scale_words

        if claim_scale and not cell_scale:
            # The claim says "1.5 thousand" and the cell says "1500".
            # The magnitudes agree once the scale word is applied, but
            # the cell carries no unit at all, so the table cannot say
            # which reading is meant. Calling this contradicted would
            # be a false accusation, so the verifier abstains.
            scaled = claim.value * UNIT_MULTIPLIERS.get(
                (claim.unit or "").strip().lower(), 1.0
            )
            if abs(scaled - value) <= max(1.0, abs(scaled) * 0.01):
                return NumericFinding(
                    claim=claim,
                    verdict=Verdict.UNCERTAIN,
                    explanation=(
                        f"the cell reads {raw!r} with no unit, which matches the"
                        f" claim once the scale word is applied, but the table"
                        f" does not state the unit, so the match is not settled"
                    ),
                    cells_used=[cell],
                    arithmetic=(
                        f"table {value} (no unit) vs claim {claim.value}"
                        f" {claim.unit} -> {scaled}"
                    ),
                )

        if claim.unit and unit and claim.unit not in unit and unit not in claim.unit:
            # The claim and the cell disagree about magnitude words.
            multiplier_claim = UNIT_MULTIPLIERS.get(claim.unit, 1.0)
            expected = _normalise(value, unit) * multiplier_claim
            if abs(expected - claim.value) > max(1.0, abs(claim.value) * 0.01):
                return NumericFinding(
                    claim=claim,
                    verdict=Verdict.CONTRADICTED,
                    explanation=(
                        f"the cell reads {raw!r} but the claim says {claim.value} "
                        f"{claim.unit}"
                    ),
                    cells_used=[cell],
                    arithmetic=f"table {value} {unit} -> {expected}",
                )

        if abs(expected - claim.value) <= max(0.01, abs(expected) * 0.005):
            return NumericFinding(
                claim=claim,
                verdict=Verdict.SUPPORTED,
                explanation=(
                    f"the cell at row {row_index} column {column_index} reads {raw!r}"
                ),
                cells_used=[cell],
                arithmetic=f"{value} {unit} = {expected}, claim says {claim.value}",
            )

        return NumericFinding(
            claim=claim,
            verdict=Verdict.CONTRADICTED,
            explanation=(
                f"the cell at row {row_index} column {column_index} reads {raw!r}, "
                f"not {claim.value}"
            ),
            cells_used=[cell],
            arithmetic=f"{value} {unit} = {expected}, claim says {claim.value}",
        )

    if claim.operation == "percent_change":
        if row_index < 0:
            return NumericFinding(
                claim=claim,
                verdict=Verdict.UNCERTAIN,
                explanation="a percentage change needs a row to compare",
            )
        earlier, later = _comparison_columns(table, claim)
        if earlier < 0 or later < 0:
            return NumericFinding(
                claim=claim,
                verdict=Verdict.UNCERTAIN,
                explanation=(
                    "the table does not have two period columns to compare"
                ),
            )
        column_index = later
        if earlier < 0:
            return NumericFinding(
                claim=claim,
                verdict=Verdict.UNCERTAIN,
                explanation=(
                    "the table does not have an earlier period column to compare against"
                ),
            )
        old_raw = table.cell(row_index, earlier)
        new_raw = table.cell(row_index, column_index)
        old_value, old_unit = _parse_value(old_raw)
        new_value, new_unit = _parse_value(new_raw)
        if old_value in (None, 0) or new_value is None:
            return NumericFinding(
                claim=claim,
                verdict=Verdict.UNCERTAIN,
                explanation="the comparison cells are not usable numbers",
            )
        change = (new_value - old_value) / abs(old_value) * 100.0
        cells = [
            {
                "row_label": table.row_labels[row_index],
                "column_header": table.headers[earlier],
                "raw": old_raw,
                "value": old_value,
                "unit": old_unit,
            },
            {
                "row_label": table.row_labels[row_index],
                "column_header": table.headers[column_index],
                "raw": new_raw,
                "value": new_value,
                "unit": new_unit,
            },
        ]
        arithmetic = (
            f"({new_value} - {old_value}) / {old_value} * 100 = {change:.1f}%"
        )
        if claim.operation_value is None:
            return NumericFinding(
                claim=claim,
                verdict=Verdict.UNCERTAIN,
                explanation="the claim does not state the percentage change",
                cells_used=cells,
                arithmetic=arithmetic,
            )
        if abs(change - claim.operation_value) <= max(0.1, abs(change) * 0.01):
            return NumericFinding(
                claim=claim,
                verdict=Verdict.SUPPORTED,
                explanation=f"the computed change matches the claimed {claim.operation_value}%",
                cells_used=cells,
                arithmetic=arithmetic,
            )
        return NumericFinding(
            claim=claim,
            verdict=Verdict.CONTRADICTED,
            explanation=(
                f"the computed change is {change:.1f}%, not {claim.operation_value}%"
            ),
            cells_used=cells,
            arithmetic=arithmetic,
        )

    if claim.operation == "difference":
        if row_index < 0:
            return NumericFinding(
                claim=claim,
                verdict=Verdict.UNCERTAIN,
                explanation="a difference needs a row to compare",
            )
        earlier, later = _comparison_columns(table, claim)
        if earlier < 0 or later < 0:
            return NumericFinding(
                claim=claim,
                verdict=Verdict.UNCERTAIN,
                explanation="the table does not have two period columns to compare",
            )
        column_index = later
        old_raw = table.cell(row_index, earlier)
        new_raw = table.cell(row_index, column_index)
        old_value, old_unit = _parse_value(old_raw)
        new_value, new_unit = _parse_value(new_raw)
        if old_value is None or new_value is None:
            return NumericFinding(
                claim=claim,
                verdict=Verdict.UNCERTAIN,
                explanation="the comparison cells are not usable numbers",
            )
        difference = new_value - old_value
        cells = [
            {
                "row_label": table.row_labels[row_index],
                "column_header": table.headers[earlier],
                "raw": old_raw,
                "value": old_value,
                "unit": old_unit,
            },
            {
                "row_label": table.row_labels[row_index],
                "column_header": table.headers[column_index],
                "raw": new_raw,
                "value": new_value,
                "unit": new_unit,
            },
        ]
        arithmetic = f"{new_value} - {old_value} = {difference}"
        if claim.value is None:
            return NumericFinding(
                claim=claim,
                verdict=Verdict.UNCERTAIN,
                explanation="the claim does not state the difference to compare",
                cells_used=cells,
                arithmetic=arithmetic,
            )
        if abs(difference - claim.value) <= max(0.01, abs(difference) * 0.005):
            return NumericFinding(
                claim=claim,
                verdict=Verdict.SUPPORTED,
                explanation=f"the computed difference matches the claimed {claim.value}",
                cells_used=cells,
                arithmetic=arithmetic,
            )
        return NumericFinding(
            claim=claim,
            verdict=Verdict.CONTRADICTED,
            explanation=f"the computed difference is {difference}, not {claim.value}",
            cells_used=cells,
            arithmetic=arithmetic,
        )

    return NumericFinding(
        claim=claim,
        verdict=Verdict.UNCERTAIN,
        explanation="the claim's operation could not be determined",
    )
