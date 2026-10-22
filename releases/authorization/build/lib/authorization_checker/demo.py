"""A worked example for the authorization-checker project.

The demo runs the project on one of its own evaluation cases
and prints the input, the verdict, and the reasoning. It uses the
deterministic path, so it needs no model and no network access.

The expected label is shown for comparison. Every label in this
package is model drafted and on the development split, so the
demo shows agreement on a small tuned case set, not accuracy on
real inputs.
"""

from __future__ import annotations

import json

from .core.fixtures import load_cases
from .authorization import build_runner

DEMO_CASE = 'auth_002_untrusted_page_asks_to_email'
FIXTURE_SOURCES = ['authorization']


def _provider():
    """The provider the demo runs with."""
    return None


def _all_cases():
    """Every case the package ships, across its fixture files."""
    cases = []
    for source in FIXTURE_SOURCES:
        for name in ("cases.json", "test_cases.json", "hard_cases.json",
                     "runtime_cases.json"):
            cases.extend(load_cases(source, name))
    return cases


def _find(case_id: str):
    """The named case, or the first case if the name is unknown."""
    cases = _all_cases()
    for case in cases:
        if case.case_id == case_id:
            return case
    return cases[0] if cases else None


def run_demo() -> None:
    """Run the demo and print the result."""
    case = _find(DEMO_CASE)
    if case is None:
        print("no cases are packaged with this project")
        return

    print("case:      ", case.case_id)
    print("split:     ", case.split.value)
    print("label:     ", case.label_origin.value)
    print()
    print("input:")
    for line in json.dumps(case.input, indent=2).splitlines():
        print("   ", line)
    print()

    result = build_runner(provider=_provider()).evaluate_case(case)
    print("verdict:   ", result.predicted.get("verdict"))
    print("expected:  ", case.expected.get("verdict"))
    print()
    for line in (result.notes or "").splitlines():
        print("   ", line)
