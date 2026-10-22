"""Tests for the instructions project.

These challenge behaviour rather than restating the
implementation. The cases that matter are the ones where a
plausible implementation would be wrong: a false accusation, a
false alarm, a qualifier that changes the answer, and a failure
that must not be silently turned into a verdict.
"""

from __future__ import annotations

from ai_reliability.projects.instructions import (
    check_compliance,
    parse_instructions,
)

def test_unchanged_file_predicate_detects_a_modification():
    report = check_compliance(
        ["Do not modify config/settings.toml"],
        {"config/settings.toml": "a"},
        {"config/settings.toml": "b"},
    )
    assert len(report.violated) == 1


def test_ambiguous_instruction_is_inconclusive():
    predicates, unparsed = parse_instructions(["Make the code as good as you can."])
    assert predicates == []
    assert len(unparsed) == 1


def test_missing_test_file_is_a_violation():
    report = check_compliance(
        ["Add a test for tests/test_parser.py"],
        {"src/parser.py": "x"},
        {"src/parser.py": "x"},
    )
    assert len(report.violated) == 1


def test_parser_recognises_at_least_eight_measurable_instruction_forms():
    """The parser must cover a spread of real repository instruction shapes."""
    forms = [
        "Do not modify config/settings.toml",
        "Add a test for tests/test_parser.py",
        "Create docs/guide.md",
        "Keep the output format REPORT_JSON",
        "Run the linter",
        "Remove tmp/scratch.txt",
        "Do not add new requirements*.txt",
        "Keep src/app.py under 5 lines",
        "Do not hard-code API keys",
    ]
    predicates, unparsed = parse_instructions(forms)
    kinds = {p.kind for p in predicates}
    assert unparsed == [], "every listed form must be recognised"
    assert len(kinds) >= 8, "at least eight distinct predicate kinds must be supported"


def test_unrecognised_instruction_is_reported_as_inconclusive_not_dropped():
    report = check_compliance(
        ["Make the code as good as you can."],
        {"src/app.py": "x"},
        {"src/app.py": "y"},
    )
    assert report.unparsed_instructions
    assert not report.violated


def test_removed_file_predicate_flags_a_file_that_survived():
    report = check_compliance(
        ["Remove tmp/scratch.txt"],
        {"tmp/scratch.txt": "x", "src/app.py": "y"},
        {"tmp/scratch.txt": "x", "src/app.py": "y"},
    )
    assert len(report.violated) == 1
    assert "scratch.txt" in report.violated[0].detail


def test_forbidden_new_file_predicate_ignores_pre_existing_files():
    """A file that was already there is not a violation of 'do not add'."""
    report = check_compliance(
        ["Do not add new requirements*.txt"],
        {"docs/requirements_old.txt": "x"},
        {"docs/requirements_old.txt": "x"},
    )
    assert not report.violated


def test_forbidden_new_file_predicate_flags_a_new_matching_file():
    report = check_compliance(
        ["Do not add new requirements*.txt"],
        {"src/app.py": "x"},
        {"src/app.py": "x", "docs/requirements_extra.txt": "y"},
    )
    assert len(report.violated) == 1


def test_line_limit_predicate_measures_the_actual_line_count():
    report = check_compliance(
        ["Keep src/app.py under 3 lines"],
        {"src/app.py": "x"},
        {"src/app.py": "a\nb\nc\nd\n"},
    )
    assert len(report.violated) == 1
    assert "4 lines" in report.violated[0].evidence


def test_secret_predicate_flags_a_credential_in_a_final_artifact():
    report = check_compliance(
        ["Do not hard-code API keys"],
        {"src/app.py": "x"},
        {"src/app.py": "API_KEY = \"sk-live-abc123\"\n"},
    )
    assert len(report.violated) == 1


def test_secret_predicate_does_not_fire_on_ordinary_code():
    """The secret check must not become a false-accusation generator."""
    report = check_compliance(
        ["Do not hard-code API keys"],
        {"src/app.py": "x"},
        {"src/app.py": "def get_key():\n    return os.environ['KEY']\n"},
    )
    assert not report.violated
