"""Tests for the release builder.

The rule a release has to satisfy is that it stands alone. These tests
check the parts of the builder that could quietly break that: the
import rewriting, the core module closure, and the verifier that
catches a release reaching back into the lab.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from ai_reliability.release import (
    AGGREGATORS,
    DISTRIBUTION_NAMES,
    _module_imports,
    _rewrite_imports,
    _rewrite_test_imports,
    list_plans,
    package_dir_name,
    plan_release,
    verify_release,
)
from ai_reliability.release_content import CONTENT


def test_every_project_has_written_content():
    """A project with no README content is not release ready."""
    for plan in list_plans():
        if plan.project in AGGREGATORS:
            continue
        assert plan.project in CONTENT, plan.project
        assert CONTENT[plan.project].summary
        assert CONTENT[plan.project].question
        assert CONTENT[plan.project].why
        assert CONTENT[plan.project].rules
        assert CONTENT[plan.project].limits


def test_the_aggregator_project_is_not_released():
    """The regression suite needs the projects it measures."""
    plan = plan_release("eval_regression")
    assert plan.releasable is False
    assert plan.reason
    assert "eval_regression" in AGGREGATORS


def test_every_other_project_is_releasable():
    for plan in list_plans():
        if plan.project in AGGREGATORS:
            continue
        assert plan.releasable, "%s: %s" % (plan.project, plan.reason)
        assert plan.core_modules
        assert plan.fixture_files or plan.borrowed_fixtures


def test_no_project_reaches_back_into_another_project():
    """A project that imported another project could not be released."""
    for plan in list_plans():
        if plan.project in AGGREGATORS:
            continue
        assert not plan.sibling_projects, plan.project


def test_the_core_closure_includes_transitive_dependencies():
    """A core module that needs another core module must bring it."""
    plan = plan_release("claims")
    # claims needs documents, which needs sources and spans.
    assert "documents" in plan.core_modules
    assert "sources" in plan.core_modules
    assert "spans" in plan.core_modules
    assert "types" in plan.core_modules


def test_a_project_module_import_becomes_a_release_import():
    """The two level lab import becomes a one level release import."""
    text = "from ..core.types import Verdict\nfrom ..core.retrieval import retrieve\n"
    rewritten = _rewrite_imports(text)
    assert "from .core.types import Verdict" in rewritten
    assert "from .core.retrieval import retrieve" in rewritten
    assert "..core" not in rewritten


def test_a_test_file_import_becomes_a_package_import():
    """A test file is not in the package, so it names it absolutely."""
    plan = plan_release("claims")
    text = (
        "from ai_reliability.core.providers import FakeProvider\n"
        "from ai_reliability.projects.claims import audit_claims\n"
    )
    rewritten = _rewrite_test_imports(text, plan)
    package = package_dir_name("claims")
    assert "from %s.core.providers import FakeProvider" % package in rewritten
    assert "from %s.claims import audit_claims" % package in rewritten
    assert "ai_reliability" not in rewritten


def test_sibling_imports_are_read_in_context():
    """A one level import means a sibling, and what a sibling is depends
    on which package the file is in."""
    from ai_reliability.core import verdicts as core_verdicts
    from ai_reliability.projects import claims as project_claims

    core_reading = _module_imports(Path(core_verdicts.__file__), "core")
    project_reading = _module_imports(Path(project_claims.__file__), "project")
    assert "core:types" in core_reading
    assert not any(item.startswith("project:") for item in project_reading)


def test_the_verifier_catches_a_release_that_reaches_back():
    """The independence check has to actually fail when it should."""
    import tempfile

    with tempfile.TemporaryDirectory() as workspace:
        target = Path(workspace)
        (target / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (target / "README.md").write_text("# x\n", encoding="utf-8")
        (target / "CHANGELOG.md").write_text("# x\n", encoding="utf-8")
        (target / "LICENSE").write_text("MIT\n", encoding="utf-8")
        (target / "leaky.py").write_text(
            "from ai_reliability.core.types import Verdict\n", encoding="utf-8"
        )
        report = verify_release(target)
        assert report["independent"] is False
        assert report["problems"]


def test_the_verifier_catches_a_two_level_import():
    import tempfile

    with tempfile.TemporaryDirectory() as workspace:
        target = Path(workspace)
        (target / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (target / "README.md").write_text("# x\n", encoding="utf-8")
        (target / "CHANGELOG.md").write_text("# x\n", encoding="utf-8")
        (target / "LICENSE").write_text("MIT\n", encoding="utf-8")
        (target / "deep.py").write_text(
            "from ..core.types import Verdict\n", encoding="utf-8"
        )
        report = verify_release(target)
        assert report["independent"] is False


def test_a_mentioned_name_in_a_docstring_is_not_an_import():
    """A reference in prose is not a dependency."""
    import tempfile

    with tempfile.TemporaryDirectory() as workspace:
        target = Path(workspace)
        (target / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (target / "README.md").write_text("# x\n", encoding="utf-8")
        (target / "CHANGELOG.md").write_text("# x\n", encoding="utf-8")
        (target / "LICENSE").write_text("MIT\n", encoding="utf-8")
        (target / "doc.py").write_text(
            '"""See :mod:`ai_reliability.core.types` for details."""\n',
            encoding="utf-8",
        )
        report = verify_release(target)
        assert report["independent"] is True


def test_every_release_has_a_distribution_name():
    for plan in list_plans():
        assert plan.package == DISTRIBUTION_NAMES.get(
            plan.project, plan.project.replace("_", "-")
        )
        assert "-" in plan.package
        assert "_" not in plan.package


def test_the_package_directory_name_can_be_imported():
    """A distribution is named with hyphens, its directory with
    underscores, because that is what Python can import."""
    for plan in list_plans():
        directory = package_dir_name(plan.project)
        assert directory.isidentifier()
        assert directory.replace("_", "") == plan.package.replace("-", "")


def test_the_release_check_uses_an_absolute_path(tmp_path, monkeypatch):
    """The check runs the release with its own directory as the working
    directory, so a relative path on the path would resolve against
    the wrong place and the import would fail for the wrong reason."""
    from ai_reliability.release_verify import check_release

    # A release directory that is reached through a relative path.
    nested = tmp_path / "releases" / "claims"
    nested.mkdir(parents=True)
    (nested / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (nested / "README.md").write_text("# x\n", encoding="utf-8")
    (nested / "CHANGELOG.md").write_text("# x\n", encoding="utf-8")
    (nested / "LICENSE").write_text("MIT\n", encoding="utf-8")
    package = nested / "src" / "claim_auditor"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""x"""\n', encoding="utf-8")
    (package / "cli.py").write_text(
        "def main(argv=None):\n    return 0\n", encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    check = check_release(Path("releases") / "claims")
    assert check.imports_ok, check.imports_output
    assert check.demo_ok, check.demo_output
