"""Tests for the numerical project.

These challenge behaviour rather than restating the
implementation. The cases that matter are the ones where a
plausible implementation would be wrong: a false accusation, a
false alarm, a qualifier that changes the answer, and a failure
that must not be silently turned into a verdict.
"""

from __future__ import annotations

from ai_reliability.projects.numerical import (
    parse_table,
    verify_numeric_claim,
)

CSV = "Entity,Metric,2023,2024\nAcme Corp,Revenue,4200000,5100000\nGlobex Inc,Revenue,3800000,3900000"


def test_numerical_reads_the_right_cell():
    table = parse_table(CSV, fmt="csv")
    finding = verify_numeric_claim("Acme Corp revenue in 2024 was 5100000.", table)
    assert finding.verdict.value == "supported"
    assert finding.cells_used


def test_numerical_catches_a_wrong_year():
    table = parse_table(CSV, fmt="csv")
    finding = verify_numeric_claim("Acme Corp revenue in 2023 was 5100000.", table)
    assert finding.verdict.value == "contradicted"


def test_numerical_catches_a_wrong_row():
    table = parse_table(CSV, fmt="csv")
    finding = verify_numeric_claim("Globex Inc revenue in 2024 was 5100000.", table)
    assert finding.verdict.value == "contradicted"


def test_numerical_abstains_on_an_ambiguous_claim():
    table = parse_table(CSV, fmt="csv")
    finding = verify_numeric_claim("Revenue was 5100000.", table)
    assert finding.verdict.value in ("uncertain", "insufficient_evidence")


def test_numerical_shows_the_arithmetic():
    table = parse_table(CSV, fmt="csv")
    finding = verify_numeric_claim(
        "Acme Corp revenue increased by 21.4 percent from 2023 to 2024.", table
    )
    assert finding.verdict.value == "supported"
    assert "100" in finding.arithmetic
