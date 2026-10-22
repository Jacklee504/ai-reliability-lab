"""Tests for the code_review project.

These challenge behaviour rather than restating the
implementation. The cases that matter are the ones where a
plausible implementation would be wrong: a false accusation, a
false alarm, a qualifier that changes the answer, and a failure
that must not be silently turned into a verdict.
"""

from __future__ import annotations

from code_review_verifier.code_review import (
    CodeSnapshot,
    ReviewComment,
    extract_assertion,
    verify_review_comment,
    verify_runtime_claim,
)

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
    from code_review_verifier.core.execution import run_harness

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
