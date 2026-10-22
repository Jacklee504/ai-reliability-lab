"""Fixture loading.

Fixtures are stored as JSON files under the ``fixtures`` directory, one
sub-directory per project, so they can be inspected, diffed, and
hand-edited. Each fixture file holds a list of cases; each case carries
its own label origin so the evaluation report can separate human-verified
labels from synthetic or model-drafted ones.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .types import EvalCase, LabelOrigin, Split

FIXTURES_DIRNAME = "fixtures"


def fixtures_root() -> Path:
    """Locate the fixtures directory.

    Resolution order: the ``AIRL_FIXTURES`` environment variable, then
    the fixtures directory shipped inside the package, then a fixtures
    directory found by walking up from this file, then the current
    working directory.

    The packaged directory comes first because an installed package
    has no checkout to walk up to. Without it, a wheel would install
    cleanly and then fail to find its own cases.
    """
    env = os.environ.get("AIRL_FIXTURES")
    if env:
        return Path(env).expanduser()

    here = Path(__file__).resolve()
    packaged = here.parent.parent / FIXTURES_DIRNAME
    if packaged.is_dir():
        return packaged
    for parent in here.parents:
        candidate = parent / FIXTURES_DIRNAME
        if candidate.is_dir():
            return candidate
    return Path.cwd() / FIXTURES_DIRNAME


def project_fixture_dir(project: str) -> Path:
    """Return the fixture directory for one project."""
    return fixtures_root() / project


def load_cases(project: str, filename: str | None = None) -> list[EvalCase]:
    """Load evaluation cases for a project.

    ``filename`` defaults to ``cases.json`` in the project's fixture
    directory.
    """
    target = project_fixture_dir(project) / (filename or "cases.json")
    if not target.exists():
        return []
    payload = json.loads(target.read_text(encoding="utf-8"))
    entries = payload.get("cases", []) if isinstance(payload, dict) else payload
    cases: list[EvalCase] = []
    for entry in entries:
        cases.append(
            EvalCase(
                case_id=entry["case_id"],
                project=entry.get("project", project),
                input=entry.get("input", {}),
                expected=entry.get("expected", {}),
                label_origin=LabelOrigin(entry.get("label_origin", "unknown")),
                split=Split(entry.get("split", "test")),
                rationale=entry.get("rationale", ""),
                tags=tuple(entry.get("tags", [])),
                notes=entry.get("notes"),
            )
        )
    return cases


def load_project_cases(project: str) -> list[EvalCase]:
    """Load every case a project owns, across its splits.

    A project may keep its development cases in ``cases.json`` and a
    genuinely held-out set in ``test_cases.json``. The held-out file
    is loaded here so that a project does not have to know about file
    names, and so the split recorded on each case is the one the case
    actually came from.

    The held-out file is *not* tuned on. The evaluation reports the
    split sizes so it is visible which cases were used for development
    and which were only ever scored once.
    """
    cases = load_cases(project)
    test_file = project_fixture_dir(project) / "test_cases.json"
    if not test_file.exists():
        return cases
    for case in load_cases(project, "test_cases.json"):
        if case.split is Split.TEST:
            cases.append(case)
            continue
        cases.append(
            EvalCase(
                case_id=case.case_id,
                project=case.project,
                input=case.input,
                expected=case.expected,
                label_origin=case.label_origin,
                split=Split.TEST,
                rationale=case.rationale,
                tags=case.tags,
                notes=case.notes,
            )
        )
    return cases


def load_json(project: str, filename: str) -> Any:
    """Load an arbitrary JSON fixture file for a project."""
    target = project_fixture_dir(project) / filename
    return json.loads(target.read_text(encoding="utf-8"))


def dump_cases(cases: list[EvalCase], path: str | os.PathLike[str]) -> Path:
    """Write cases back out in the fixture format."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cases": [
            {
                "case_id": c.case_id,
                "project": c.project,
                "input": c.input,
                "expected": c.expected,
                "label_origin": c.label_origin.value,
                "split": c.split.value,
                "rationale": c.rationale,
                "tags": list(c.tags),
                **({"notes": c.notes} if c.notes else {}),
            }
            for c in cases
        ]
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target
