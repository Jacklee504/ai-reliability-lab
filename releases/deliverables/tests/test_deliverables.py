"""Tests for the deliverables project.

These challenge behaviour rather than restating the
implementation. The cases that matter are the ones where a
plausible implementation would be wrong: a false accusation, a
false alarm, a qualifier that changes the answer, and a failure
that must not be silently turned into a verdict.
"""

from __future__ import annotations

import json

from deliverable_checker.deliverables import (
    Requirement,
    check_deliverable,
    extract_requirements,
)

def test_requirement_extraction_does_not_invent_requirements():
    requirements = extract_requirements("Make it good.")
    assert requirements == []


def test_requirement_extraction_understands_number_words():
    requirements = extract_requirements("Write 50 questions across six chapters.")
    counts = {r.count for r in requirements if r.kind == "count"}
    assert 50 in counts
    assert 6 in counts


def test_deliverable_with_no_requirements_cannot_verify():
    report = check_deliverable("Make it good.", "It is good now.")
    assert report.verdict == "cannot_verify"


def test_deliverable_missing_section_fails():
    report = check_deliverable(
        "Write 2 questions across 2 chapters, four options each, with an answer key.",
        "CHAPTER 1\nQ1) What is 2+2?\nA) 3\nB) 4\nC) 5\nD) 6\n",
    )
    assert report.verdict == "failed"


def test_edited_requirement_list_is_used_for_recheck():
    request = "Write two questions."
    requirements = extract_requirements(request)
    assert requirements
    requirements[0].count = 1
    report = check_deliverable(request, "Q1) One?", requirements=requirements)
    assert report.requirements[0].requirement_id == requirements[0].requirement_id
    assert report.checks[0].requirement.requirement_id == requirements[0].requirement_id


def test_exact_count_does_not_accept_extra_questions():
    material = "Q1) One?\nQ2) Two?\nQ3) Three?"
    assert check_deliverable("Write 2 questions.", material).verdict == "failed"
    assert check_deliverable("Write at least 2 questions.", material).verdict == "passed"
    assert check_deliverable("Write at most 2 questions.", material).verdict == "failed"


def test_options_are_checked_for_every_question():
    material = "Q1) One?\nA) a\nB) b\nC) c\nD) d\nQ2) Two?\nA) a\nB) b\nC) c"
    report = check_deliverable("Write 2 questions with four options each.", material)
    assert report.verdict == "failed"


def test_edited_requirement_ids_survive_the_recheck():
    """A user-edited requirement list must keep its IDs through a recheck."""
    from deliverable_checker.deliverables import Requirement, check_deliverable

    edited = [
        {
            "requirement_id": "req_01",
            "text": "must contain 2 question",
            "kind": "count",
            "check": "deterministic",
            "target": "",
            "count": 2,
            "unit": "question",
            "source_quote": "Write two questions.",
            "note": "edited by the reviewer",
            "comparator": "exactly",
        }
    ]
    report = check_deliverable(
        "Write two questions.",
        "Q1) 2+2?\nA) 3\nB) 4\n",
        requirements=[Requirement.from_dict(entry) for entry in edited],
    )
    assert [r.requirement_id for r in report.requirements] == ["req_01"]
    assert report.checks[0].requirement.requirement_id == "req_01"
    assert report.checks[0].requirement.note == "edited by the reviewer"


def test_requirement_parsing_ignores_unknown_edited_fields():
    from deliverable_checker.deliverables import Requirement

    requirement = Requirement.from_dict({
        "requirement_id": "req_01",
        "text": "must contain 2 question",
        "kind": "count",
        "check": "deterministic",
        "count": 2,
        "unit": "question",
        "reviewer_annotation": "checked by hand",
    })
    assert requirement.requirement_id == "req_01"
    assert requirement.count == 2
    assert requirement.comparator == "exactly"
