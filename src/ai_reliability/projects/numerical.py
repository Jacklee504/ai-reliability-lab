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
