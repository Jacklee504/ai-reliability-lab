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


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------


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


def test_edited_requirement_ids_survive_the_recheck():
    """A user-edited requirement list must keep its IDs through a recheck."""
    from ai_reliability.projects.deliverables import Requirement, check_deliverable

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
    from ai_reliability.projects.deliverables import Requirement

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

# ---------------------------------------------------------------------------
# Code review
# ---------------------------------------------------------------------------


GUARDED = "def f(x):\n    if x is None:\n        raise ValueError()\n    return x\n"
UNGUARDED = "def f(x):\n    return x\n"


def test_review_comment_is_refuted_when_a_guard_exists():
    snapshot = CodeSnapshot(identifier="s", files={"a.py": GUARDED})
    finding = verify_review_comment(
        ReviewComment(text="There is no null check on x.", file="a.py", line=1),
        snapshot,
    )
    assert finding.verdict.value == "contradicted"


def test_runtime_exception_comment_abstains_without_execution():
    snapshot = CodeSnapshot(identifier="s", files={"a.py": UNGUARDED})
    finding = verify_review_comment(
        ReviewComment(text="This will raise an error.", file="a.py", line=1),
        snapshot,
    )
    assert finding.verdict.value == "uncertain"


def test_guard_in_another_function_does_not_refute_a_missing_guard_comment():
    code = "def other(x):\n    if x is None:\n        raise ValueError()\n\ndef target(y):\n    return y\n"
    snapshot = CodeSnapshot(identifier="s", files={"a.py": code})
    finding = verify_review_comment(
        ReviewComment(text="There is no null check on y.", file="a.py", line=5),
        snapshot,
    )
    assert finding.verdict.value == "supported"


def test_guard_after_query_does_not_refute_before_query_comment():
    code = "def f(cursor, x):\n    cursor.execute(x)\n    if x is None:\n        raise ValueError()\n"
    snapshot = CodeSnapshot(identifier="s", files={"a.py": code})
    finding = verify_review_comment(
        ReviewComment(text="There is no null check on x before the query runs.", file="a.py", line=1),
        snapshot,
    )
    assert finding.verdict.value == "supported"


def test_malformed_code_does_not_support_a_missing_guard_accusation():
    snapshot = CodeSnapshot(identifier="s", files={"a.py": "def f(x):\n    if x is None\n"})
    finding = verify_review_comment(
        ReviewComment(text="There is no null check on x.", file="a.py", line=1), snapshot,
    )
    assert finding.verdict.value == "uncertain"


def test_runtime_claim_is_confirmed_by_actually_running_the_harness():
    code = "def divide(a, b):\n    return a / b\n"
    snapshot = CodeSnapshot(identifier="s", files={"math_ops.py": code})
    finding = verify_runtime_claim(
        ReviewComment(
            text="Calling this with a zero denominator will raise an error.",
            file="math_ops.py",
            line=1,
        ),
        snapshot,
        "import math_ops\nmath_ops.divide(1, 0)\n",
    )
    assert finding.verdict.value == "supported"
    assert finding.method.value == "executed"
    assert finding.execution is not None
    assert finding.execution.raised is True


def test_runtime_claim_is_refuted_when_the_harness_does_not_raise():
    code = "def divide(a, b):\n    if b == 0:\n        return 0\n    return a / b\n"
    snapshot = CodeSnapshot(identifier="s", files={"math_ops.py": code})
    finding = verify_runtime_claim(
        ReviewComment(
            text="Calling this with a zero denominator will raise an error.",
            file="math_ops.py",
            line=1,
        ),
        snapshot,
        "import math_ops\nmath_ops.divide(1, 0)\n",
    )
    assert finding.verdict.value == "contradicted"


def test_broken_harness_is_unverified_rather_than_evidence():
    """A harness that cannot import must not manufacture a verdict."""
    snapshot = CodeSnapshot(identifier="s", files={"app.py": "def go():\n    return 1\n"})
    finding = verify_runtime_claim(
        ReviewComment(text="This will raise an error.", file="app.py", line=1),
        snapshot,
        "import module_that_does_not_exist_xyz\n",
    )
    assert finding.verdict.value == "uncertain"


def test_runtime_claim_without_a_harness_stays_unverified():
    snapshot = CodeSnapshot(identifier="s", files={"app.py": "def go():\n    return 1\n"})
    finding = verify_runtime_claim(
        ReviewComment(text="This will raise an error.", file="app.py", line=1),
        snapshot,
        "",
    )
    assert finding.verdict.value == "uncertain"


def test_a_hanging_harness_is_unverified_not_a_verdict():
    """A timeout produces no observation, so no verdict may be claimed."""
    snapshot = CodeSnapshot(
        identifier="s", files={"spin.py": "def spin():\n    while True:\n        pass\n"}
    )
    finding = verify_runtime_claim(
        ReviewComment(text="This will raise an error.", file="spin.py", line=1),
        snapshot,
        "import spin\nspin.spin()\n",
        timeout=2.0,
    )
    assert finding.verdict.value == "uncertain"
    assert finding.execution is not None
    assert finding.execution.timed_out is True


def test_the_harness_cannot_see_the_surrounding_workspace():
    """Execution must be isolated from the machine it runs on."""
    from ai_reliability.core.execution import run_harness

    outcome = run_harness(
        {"mod.py": "x = 1\n"},
        "import ai_reliability\n",
    )
    assert outcome.observed is False
    assert "harness itself failed" in (outcome.error or "")


def test_out_of_range_review_location_is_unverified():
    snapshot = CodeSnapshot(identifier="s", files={"a.py": UNGUARDED})
    finding = verify_review_comment(
        ReviewComment(text="There is no null check on x.", file="a.py", line=50), snapshot,
    )
    assert finding.verdict.value == "insufficient_evidence"


def test_review_comment_on_a_missing_file_is_unverified():
    snapshot = CodeSnapshot(identifier="s", files={"a.py": UNGUARDED})
    finding = verify_review_comment(
        ReviewComment(text="There is no null check.", file="other.py", line=1),
        snapshot,
    )
    assert finding.verdict.value in ("uncertain", "insufficient_evidence")


def test_assertion_extraction_drops_preamble():
    assertion = extract_assertion("Nit: This will raise an error on empty input.")
    assert assertion.startswith("This will raise")


def test_source_edit_only_reopens_overlapping_claim():
    report = audit_versions(
        "Fee: 10\nNotice: 30\n", "Fee: 20\nNotice: 30\n",
        [EvidenceReference("fee", 0, 7), EvidenceReference("notice", 8, 18)],
    )
    assert report["revisit_claim_ids"] == ["fee"]
    assert report["conclusion"].startswith("changed evidence requires review")


def test_source_hash_mismatch_fails_closed_into_review_queue():
    report = audit_versions("A\n", "B\n", [EvidenceReference("c", 0, 1, "wrong")])
    assert report["revisit_claim_ids"] == ["c"]
    assert report["warnings"]


def test_study_paraphrase_does_not_become_an_omission_claim():
    report = audit_coverage(
        [{"id": "recursion", "label": "recursion"}],
        "A function calls itself to solve smaller problems.",
    )
    assert report["topics"][0]["status"] == "unverified"


def test_study_counts_label_and_alias_mentions_without_double_counting():
    report = audit_coverage(
        [{"id": "pcr", "label": "polymerase chain reaction", "aliases": ["PCR"]}],
        "PCR is short for polymerase chain reaction. PCR is common.",
    )
    assert report["topics"][0]["mentions"] == 3


def test_review_comparison_reports_disagreement_without_changing_fixture():
    result = compare_review_packet({
        "cases": [{"case_id": "claim_001_supported", "reviewed_expected": {"verdict": "contradicted"}}]
    })
    assert result["reviewed_cases"] == 1
    assert result["disagreements"][0]["case_id"] == "claim_001_supported"
    assert result["unreviewed_cases"] == result["total_cases"] - 1


def test_review_comparison_accepts_verdict_only_without_requiring_fixture_metadata():
    result = compare_review_packet({
        "cases": [{"case_id": "claim_001_supported", "reviewed_expected": {"verdict": "supported"}}]
    })
    assert result["reviewed_cases"] == 1
    assert result["disagreements"] == []


def test_review_comparison_surfaces_draft_alternatives_even_when_choice_matches():
    result = compare_review_packet({
        "cases": [{"case_id": "claim_004_offtopic_source", "reviewed_expected": {"verdict": "insufficient_evidence"}}]
    })
    assert result["disagreements"] == []
    assert result["ambiguous_drafts"] == [{
        "case_id": "claim_004_offtopic_source",
        "field": "verdict",
        "draft_options": ["insufficient_evidence", "uncertain"],
    }]

def test_an_insertion_before_the_evidence_queues_the_claim():
    """An insertion above the span shifts the text the span pointed at."""
    from ai_reliability.projects.source_versions import EvidenceReference, audit_versions

    before = "SERVICE AGREEMENT\n\n1. Fees\nThe monthly fee is 2,400 EUR.\n"
    after = "SERVICE AGREEMENT\n\n0. Preamble\nThis agreement is made on 1 January.\n\n1. Fees\nThe monthly fee is 2,400 EUR.\n"
    start = before.index("The monthly fee is 2,400 EUR.\n")
    report = audit_versions(
        before, after,
        [EvidenceReference(claim_id="fee", start=start, end=start + 30)],
    )
    assert report["revisit_claim_ids"] == ["fee"]


def test_an_unrelated_edit_does_not_queue_the_claim():
    from ai_reliability.projects.source_versions import EvidenceReference, audit_versions

    before = "SERVICE AGREEMENT\n\n1. Fees\nThe monthly fee is 2,400 EUR.\n\n2. Support\nSupport is available on business days.\n"
    after = "SERVICE AGREEMENT\n\n1. Fees\nThe monthly fee is 2,400 EUR.\n\n2. Support\nSupport is available on weekdays.\n"
    start = before.index("The monthly fee is 2,400 EUR.\n")
    report = audit_versions(
        before, after,
        [EvidenceReference(claim_id="fee", start=start, end=start + 30)],
    )
    assert report["revisit_claim_ids"] == []


