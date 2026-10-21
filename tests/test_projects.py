"""Tests for the project modules.

These challenge behaviour rather than restating the implementation. The
cases that matter are the ones where a plausible implementation would be
wrong: a false accusation, a false alarm, a qualifier that changes the
answer, and a failure that must not be silently turned into a verdict.
"""

from __future__ import annotations

import json

from ai_reliability.core.providers import FakeProvider, ParseError
from ai_reliability.cli import compare_review_packet
from ai_reliability.core.types import EvalCase, LabelOrigin, Split
from ai_reliability.projects.authorization import (
    Authority,
    Authorization,
    ProposedAction,
    evaluate_action,
)
from ai_reliability.projects.baselines import BaselineRunner, run_naive_baseline
from ai_reliability.projects.citations import Citation, audit_citations
from ai_reliability.projects.claims import SourceDocument, audit_claims, build_claim_prompt
from ai_reliability.projects.code_review import (
    CodeSnapshot,
    ReviewComment,
    extract_assertion,
    verify_review_comment,
    verify_runtime_claim,
)
from ai_reliability.projects.deliverables import check_deliverable, extract_requirements
from ai_reliability.projects.instructions import (
    check_compliance,
    parse_instructions,
)
from ai_reliability.projects.numerical import (
    parse_table,
    verify_numeric_claim,
)
from ai_reliability.projects.omission import audit_omissions
from ai_reliability.projects.source_versions import EvidenceReference, audit_versions
from ai_reliability.projects.study_coverage import audit_coverage


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


def test_claim_audit_never_contradicts_without_evidence():
    """A claim with no matching source must abstain, never accuse."""
    report = audit_claims(
        "The office is located in Dublin.",
        [SourceDocument(identifier="s", text="The monthly service fee is 2,400 EUR.")],
        provider=None,
    )
    assert report.findings
    for finding in report.findings:
        assert finding.verdict.verdict.value != "contradicted"


def test_claim_audit_splits_a_multi_claim_sentence():
    """One sentence with two claims must produce two findings."""
    report = audit_claims(
        "The fee is 2,400 EUR and the notice period is 60 days.",
        [SourceDocument(identifier="s", text="The fee is 2,400 EUR. The notice period is 90 days.")],
        provider=None,
    )
    assert len(report.findings) >= 2


def test_claim_prompt_names_all_four_outcomes():
    prompt = build_claim_prompt("A claim", ["A passage"])
    for verdict in ("supported", "contradicted", "insufficient_evidence", "uncertain"):
        assert verdict in prompt


def test_claim_audit_records_a_model_parse_failure():
    """A model that returns nonsense must be recorded, not silently used."""
    provider = FakeProvider(
        responses={"claim_verdict": "this is not json at all"},
    )
    report = audit_claims(
        "The fee is 2,400 EUR.",
        [SourceDocument(identifier="s", text="The fee is 2,400 EUR.")],
        provider=provider,
    )
    assert report.parse_failures >= 1
    for finding in report.findings:
        assert finding.verdict.verdict.value != "contradicted"


def test_naive_judges_keep_nonanswers_out_of_binary_verdicts():
    binary = FakeProvider(responses={"naive_binary": "Not enough information."})
    assert run_naive_baseline(binary, "A", ["B"]).verdict is None
    three_way = FakeProvider(responses={"naive_verdict": "Not enough information."})
    assert run_naive_baseline(three_way, "A", ["B"], "naive_verdict").verdict.value == "insufficient_evidence"


def test_multi_claim_baseline_requires_the_expected_sequence():
    case = EvalCase(
        case_id="mixed", project="claims",
        input={"answer": "The fee is 10 EUR and the notice is 30 days.", "sources": []},
        expected={"verdicts": ["supported", "contradicted"]},
    )
    always_yes = FakeProvider(responses={"naive_binary": "Yes"})
    result = BaselineRunner(provider=always_yes).evaluate_case(case)
    assert result.predicted["verdicts"] == ["supported", "supported"]
    assert result.correct is False


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def test_citation_audit_reports_topical_overlap_separately():
    """Topical relevance must never be presented as support."""
    report = audit_citations(
        [Citation(claim="The fee is 2,400 EUR.", passages=["The fee is described here."])],
        provider=None,
    )
    finding = report.findings[0]
    assert finding.max_topical > 0.0
    assert finding.verdict.verdict.value != "supported"

def test_a_citation_about_a_different_entity_cannot_support_the_claim():
    """A model that bridges two named entities must not be believed."""
    provider = FakeProvider(responses={"citation_support": json.dumps({
        "verdict": "supported",
        "explanation": "The graduate programme is an early-career programme.",
        "key_phrase": "The graduate programme runs for 12 months.",
    })})
    report = audit_citations(
        [Citation(
            claim="The early-career programme runs for 12 months.",
            passages=["The graduate programme runs for 12 months."],
        )],
        provider=provider,
    )
    finding = report.findings[0]
    assert finding.verdict.verdict.value == "insufficient_evidence"
    assert "different entity" in finding.verdict.explanation


def test_a_citation_that_mentions_the_claimed_entity_is_not_downgraded():
    provider = FakeProvider(responses={"citation_support": json.dumps({
        "verdict": "supported",
        "explanation": "The passage states the duration.",
        "key_phrase": "runs for 12 months",
    })})
    report = audit_citations(
        [Citation(
            claim="The early-career programme runs for 12 months.",
            passages=["The early-career programme runs for 12 months."],
        )],
        provider=provider,
    )
    assert report.findings[0].verdict.verdict.value == "supported"


# ---------------------------------------------------------------------------
# Omission
# ---------------------------------------------------------------------------


def test_omission_audit_on_empty_source_reports_nothing():
    report = audit_omissions("", "A summary.", "Summarize the obligations.", provider=None)
    assert report.total_statements == 0
    assert report.potential_omissions == []


def test_omission_candidates_are_ordered_by_relevance():
    source = (
        "A. Definitions of terms used here.\n"
        "The processor shall notify the controller within 48 hours.\n"
    )
    report = audit_omissions(source, "", "Summarize the security obligations.", provider=None)
    uncovered = report.potential_omissions
    if len(uncovered) >= 2:
        assert uncovered[0].relevance >= uncovered[1].relevance


# ---------------------------------------------------------------------------
# Deliverables
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Numerical
# ---------------------------------------------------------------------------


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


