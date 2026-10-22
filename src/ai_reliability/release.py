"""Build a standalone, installable package for a project.

Why this is in the product
--------------------------
The lab exists to produce portfolio pieces. A piece that can only be
run from inside the lab is a module, not a release. This module emits
the standalone package for any project: its own distribution
metadata, its own copy of the shared core it needs, its own fixtures,
its own tests, and its own command line interface.

What a release contains
-----------------------

    output directory, one subdirectory per project:
      pyproject.toml
      README.md
      CHANGELOG.md
      LICENSE
      src/PKG/__init__.py
      src/PKG/core/...        the shared core modules it needs
      src/PKG/PROJECT.py      the project module
      src/PKG/cli.py          demo and eval commands
      fixtures/PROJECT/...    its evaluation cases
      tests/test_PROJECT.py

The one rule a release must satisfy
-----------------------------------
It must not import ``ai_reliability``. A release that reached back
into the lab for one helper would not be independent, and the whole
point of extracting a project is that it stands alone. The verifier
in this module checks that rule, and the build is written so the
rule holds by construction: the shared core is copied in, and the
relative imports are rewritten to point at the copy.

Deliberate limits
-----------------
* The aggregator projects, whose purpose is to measure the other
  projects against each other, are not released. They need the
  projects they measure.
* The copied core is a snapshot. A change in the lab's core has to
  be rebuilt into the releases; the build is the sync mechanism.
* The release carries the project's own fixtures and a per project
  test file, so the released package can be checked on its own.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RELEASE_VERSION = "0.1.0"

#: The projects whose purpose is to measure the other projects. They
#: are part of the lab and are not released on their own.
AGGREGATORS = {"eval_regression"}

#: Projects that run on another project's cases. The release carries
#: the borrowed fixture directory too, so the comparison still has
#: something to measure.
BORROWED_FIXTURES = {
    "baselines": ["claims", "citations"],
}

#: Projects that cannot run without a model. Their demo uses the
#: deterministic fake provider, and says so, rather than pretending the
#: offline run is the real thing.
NEEDS_MODEL = {"baselines"}

#: The distribution name for each project. A bare project name like
#: "claims" is too generic to be a distribution, and the name is what
#: a reader sees first.
DISTRIBUTION_NAMES = {
    "authorization": "authorization-checker",
    "baselines": "naive-judge-baselines",
    "citations": "citation-auditor",
    "claims": "claim-auditor",
    "code_review": "code-review-verifier",
    "context_receipts": "context-receipt",
    "deliverables": "deliverable-checker",
    "instructions": "instruction-checker",
    "numerical": "numerical-verifier",
    "omission": "omission-auditor",
    "retrieval_eval": "retrieval-eval",
    "source_versions": "source-version-tracker",
    "study_coverage": "study-coverage",
}


def _module_imports(path: Path, sibling_kind: str = "project") -> set[str]:
    """The sibling modules a file imports, as "kind:name" pairs.

    A one level relative import names a sibling in the same package.
    What that means depends on where the file lives: inside the core
    it is another core module, inside the projects package it is
    another project. The caller says which.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module:
                found.add(sibling_kind + ":" + node.module.split(".")[0])
            elif node.level == 2 and node.module:
                parts = node.module.split(".")
                if parts[0] == "core":
                    found.add("core:" + (parts[1] if len(parts) > 1 else ""))
                else:
                    found.add("project:" + parts[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("ai_reliability.core."):
                    found.add("core:" + alias.name.split(".")[2])
                elif alias.name.startswith("ai_reliability.projects."):
                    found.add("project:" + alias.name.split(".")[2])
    return found


def _core_closure(seeds: set[str], core_dir: Path) -> set[str]:
    """Every core module needed, following core to core imports."""
    needed: set[str] = set()
    frontier = set(seeds)
    while frontier:
        name = frontier.pop()
        if name in needed:
            continue
        path = core_dir / (name + ".py")
        if not path.exists():
            continue
        needed.add(name)
        for item in _module_imports(path, "core"):
            if item.startswith("core:"):
                frontier.add(item.split(":", 1)[1])
    return needed


@dataclass
class ReleasePlan:
    """What a project needs in order to stand alone."""

    project: str
    package: str
    core_modules: list[str] = field(default_factory=list)
    sibling_projects: list[str] = field(default_factory=list)
    fixture_files: list[str] = field(default_factory=list)
    borrowed_fixtures: list[str] = field(default_factory=list)
    needs_model: bool = False
    releasable: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "package": self.package,
            "core_modules": self.core_modules,
            "sibling_projects": self.sibling_projects,
            "fixture_files": self.fixture_files,
            "borrowed_fixtures": self.borrowed_fixtures,
            "needs_model": self.needs_model,
            "releasable": self.releasable,
            "reason": self.reason,
        }


def package_name(project: str) -> str:
    """The distribution name for a project."""
    return DISTRIBUTION_NAMES.get(project, project.replace("_", "-"))


def module_name(project: str) -> str:
    """The name of the project module inside the package.

    The project module keeps the lab's own name, so `claims.py` inside
    `claim_auditor` is the same file the lab runs.
    """
    return project.replace("-", "_")


def package_dir_name(project: str) -> str:
    """The importable package directory name for a project.

    The distribution is named with hyphens, the way PyPI names are
    written. The directory it installs is named with underscores,
    because that is what Python can import.
    """
    return package_name(project).replace("-", "_")


def _lab_root() -> Path:
    """The root of the installed lab package."""
    return Path(__file__).resolve().parent.parent


def _rewrite_imports(text: str) -> str:
    """Rewrite lab relative imports into release relative imports.

    Inside the lab a project module sits in the projects package and
    reaches the core with a two level relative import. In a release
    it sits beside a core subpackage, so the import becomes one
    level. Nothing else about the module changes.
    """
    return re.sub(r"^(\s*)from \.\.core\.", r"\1from .core.", text, flags=re.M)


def _rewrite_test_imports(text: str, plan: ReleasePlan) -> str:
    """Rewrite lab imports in a test file into release imports.

    A test file is not inside the package, so it cannot use relative
    imports. It names the package absolutely, and the package has a
    different name in the release.
    """
    package = package_dir_name(plan.project)
    text = text.replace("ai_reliability.core.", package + ".core.")
    text = text.replace(
        "ai_reliability.projects." + plan.project,
        package + "." + module_name(plan.project),
    )
    return text


def plan_release(project: str, lab_root: Path | None = None) -> ReleasePlan:
    """Work out what a project needs to become a standalone package."""
    root = lab_root or _lab_root()
    projects_dir = root / "ai_reliability" / "projects"
    core_dir = root / "ai_reliability" / "core"

    source = projects_dir / (project + ".py")
    if not source.exists():
        raise ValueError("unknown project %r" % project)

    core: set[str] = set()
    siblings: set[str] = set()
    for item in _module_imports(source):
        if item.startswith("core:"):
            core.add(item.split(":", 1)[1])
        elif item.startswith("project:"):
            siblings.add(item.split(":", 1)[1])
        elif item.startswith("sibling:"):
            siblings.add(item.split(":", 1)[1])

    fixtures_dir = root.parent / "fixtures" / project
    fixture_files = (
        sorted(p.name for p in fixtures_dir.glob("*.json"))
        if fixtures_dir.is_dir()
        else []
    )

    plan = ReleasePlan(
        project=project,
        package=package_name(project),
        core_modules=sorted(_core_closure(core, core_dir)),
        sibling_projects=sorted(siblings),
        fixture_files=fixture_files,
        borrowed_fixtures=list(BORROWED_FIXTURES.get(project, [])),
        needs_model=project in NEEDS_MODEL,
    )

    if project in AGGREGATORS:
        plan.releasable = False
        plan.reason = (
            "this project measures the other projects against each other, "
            "so it needs the projects it measures"
        )
    elif siblings:
        plan.releasable = False
        plan.reason = "this project imports %s" % ", ".join(sorted(siblings))
    elif not fixture_files and not plan.borrowed_fixtures:
        plan.releasable = False
        plan.reason = "this project has no evaluation fixtures of its own"
    return plan


def list_plans(lab_root: Path | None = None) -> list[ReleasePlan]:
    """Plan a release for every project in the lab."""
    root = lab_root or _lab_root()
    projects_dir = root / "ai_reliability" / "projects"
    names = sorted(
        path.stem for path in projects_dir.glob("*.py")
        if path.stem not in {"__init__", "registry"}
    )
    return [plan_release(name, root) for name in names]


def _write(path: Path, text: str) -> None:
    """Write a file, creating its parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Quality Assurance",
]


def _pyproject(plan: ReleasePlan, summary: str) -> str:
    """The distribution metadata for a release."""
    classifier_lines = "\n".join('    "%s",' % item for item in CLASSIFIERS)
    return (
        "[build-system]\n"
        'requires = ["setuptools>=68"]\n'
        'build-backend = "setuptools.build_meta"\n'
        "\n"
        "[project]\n"
        'name = "%(package)s"\n'
        'version = "%(version)s"\n'
        'description = "%(summary)s"\n'
        'readme = "README.md"\n'
        'requires-python = ">=3.11"\n'
        "license = { text = \"MIT\" }\n"
        'authors = [{ name = "Jack Lee" }]\n'
        'keywords = ["ai-reliability", "llm", "evaluation", "audit"]\n'
        "classifiers = [\n%(classifiers)s\n]\n"
        'dependencies = []\n'
        "\n"
        "[project.urls]\n"
        'Homepage = "https://github.com/JackLee504/ai-reliability-lab"\n'
        'Source = "https://github.com/JackLee504/ai-reliability-lab/tree/main/releases/%(project)s"\n'
        'Issues = "https://github.com/JackLee504/ai-reliability-lab/issues"\n'
        "\n"
        "[project.optional-dependencies]\n"
        'dev = ["pytest>=7.0"]\n'
        "\n"
        "[project.scripts]\n"
        '%(package)s = "%(module)s.cli:main"\n'
        "\n"
        "[tool.setuptools.packages.find]\n"
        'where = ["src"]\n'
        "\n"
        "[tool.setuptools.package-data]\n"
        '"%(module)s" = ["fixtures/**/*.json", "fixtures/**/*.txt",\n'
        '                "fixtures/**/*.pdf", "fixtures/**/*.bin"]\n'
        "\n"
        "[tool.pytest.ini_options]\n"
        'testpaths = ["tests"]\n'
        'pythonpath = ["src"]\n'
        'addopts = "-q"\n'
    ) % {
        "package": plan.package,
        "version": RELEASE_VERSION,
        "summary": summary,
        "module": package_dir_name(plan.project),
        "classifiers": classifier_lines,
        "project": plan.project,
    }


def _changelog(plan: ReleasePlan) -> str:
    """The changelog for a release."""
    return (
        "# Changelog\n"
        "\n"
        "All notable changes to this project are recorded here. The format\n"
        "follows Keep a Changelog, and the project uses semantic versioning.\n"
        "\n"
        "## %s\n"
        "\n"
        "### Added\n"
        "\n"
        "- First release of the %s project, extracted from the AI reliability\n"
        "  lab as a standalone package.\n"
        "- The project module, the shared core modules it needs, the project's\n"
        "  evaluation fixtures, and a per project test suite.\n"
        "- A `demo` command that runs the project on a worked example with no\n"
        "  model and no network access.\n"
        "- An `eval` command that runs the project's own evaluation cases and\n"
        "  reports agreement counts with the label provenance attached.\n"
    ) % (RELEASE_VERSION, plan.package)


def _manifest() -> str:
    """The source distribution manifest.

    The package data settings cover a wheel. An sdist needs the same
    files listed explicitly, or the cases would be missing from the
    source distribution.
    """
    return (
        "include LICENSE\n"
        "include README.md\n"
        "include CHANGELOG.md\n"
        "recursive-include src *.json *.txt *.pdf *.bin\n"
        "recursive-include fixtures *\n"
        "recursive-include tests *.py\n"
    )


def _cli_source(plan: ReleasePlan) -> str:
    """The command line interface for a release.

    Two commands: `demo` runs the project on a worked example with no
    model and no network, and `eval` runs the project's own cases.
    A reader should be able to clone the package, run the demo, and
    see the tool work before reading anything else.
    """
    sources = [plan.project] + list(plan.borrowed_fixtures)
    return (
        '"""Command line interface for the %(package)s project.\n'
        "\n"
        "Two commands:\n"
        "\n"
        "    %(package)s demo    run the project on a worked example\n"
        "    %(package)s eval    run the project's evaluation cases\n"
        "\n"
        "Both work with no model and no network access. The demo uses the\n"
        "deterministic path, so it shows the behaviour a reader can rely\n"
        "on without a model endpoint.\n"
        '"""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "import argparse\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "from .core.evaluation import run_evaluation\n"
        "from .core.fixtures import load_cases\n"
        "from .%(module)s import build_cases, build_runner\n"
        "\n"
        "FIXTURE_SOURCES = %(sources)r\n"
        "\n"
        "\n"
        "def _fixtures_dir() -> Path:\n"
        '    """Where the packaged fixtures live."""\n'
        "    here = Path(__file__).resolve().parent.parent.parent\n"
        '    return here / "fixtures" / "%(project)s"\n'
        "\n"
        "\n"
        "def cmd_demo(args: argparse.Namespace) -> int:\n"
        '    """Run the project on a worked example."""\n'
        "    from .demo import run_demo\n"
        "\n"
        "    run_demo()\n"
        "    return 0\n"
        "\n"
        "\n"
        "def cmd_eval(args: argparse.Namespace) -> int:\n"
        '    """Run the project\'s own evaluation cases."""\n'
        "    cases = []\n"
        "    for source in FIXTURE_SOURCES:\n"
        "        cases.extend(load_cases(source, args.fixture))\n"
        "    if not cases:\n"
        '        print("no cases found in", _fixtures_dir(), file=sys.stderr)\n'
        "        return 1\n"
        "    runner = build_runner(provider=None)\n"
        "    report = run_evaluation(\n"
        '        "%(project)s", cases, runner, method="deterministic"\n'
        "    )\n"
        "    if args.json:\n"
        "        print(json.dumps(report.to_dict(), indent=2, default=str))\n"
        "    else:\n"
        "        print(report.render())\n"
        "    return 0\n"
        "\n"
        "\n"
        "def build_parser() -> argparse.ArgumentParser:\n"
        "    parser = argparse.ArgumentParser(\n"
        '        prog="%(package)s",\n'
        '        description="%(summary)s",\n'
        "    )\n"
        '    sub = parser.add_subparsers(dest="command", required=True)\n'
        "\n"
        '    demo = sub.add_parser("demo", help="run the worked example")\n'
        "    demo.set_defaults(func=cmd_demo)\n"
        "\n"
        '    evaluate = sub.add_parser("eval", help="run the evaluation cases")\n'
        '    evaluate.add_argument("--fixture", default="cases.json")\n'
        '    evaluate.add_argument("--json", action="store_true")\n'
        "    evaluate.set_defaults(func=cmd_eval)\n"
        "\n"
        "    return parser\n"
        "\n"
        "\n"
        "def main(argv: list[str] | None = None) -> int:\n"
        "    parser = build_parser()\n"
        "    args = parser.parse_args(argv)\n"
        "    try:\n"
        "        return int(args.func(args) or 0)\n"
        "    except KeyboardInterrupt:\n"
        '        print("interrupted", file=sys.stderr)\n'
        "        return 130\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n"
    ) % {
        "package": plan.package,
        "module": module_name(plan.project),
        "project": plan.project,
        "summary": _first_line(plan.project),
        "sources": sources,
    }


#: The case each project demonstrates. The choice is deliberate: for
#: most projects it is a case where the tool abstains or refuses to
#: accuse, because that is the behaviour worth showing.
DEMO_CASES = {
    "authorization": "auth_002_untrusted_page_asks_to_email",
    "baselines": "claim_004_offtopic_source",
    "citations": "cit_002_keyword_match_wrong_year",
    "claims": "claim_004_offtopic_source",
    "code_review": "cr_002_false_null_warning",
    "context_receipts": "receipt_005_stale_version_detectable",
    "deliverables": "deliv_001_complete_quiz",
    "instructions": "instr_002_unchanged_file_violated",
    "numerical": "num_001_correct_read",
    "omission": "om_001_material_omission_breach_notice",
    "retrieval_eval": "ret_001_exact_terms",
    "source_versions": "sv_001_changed_fee",
    "study_coverage": "sc_001_direct_mention",
}


def _demo_source(plan: ReleasePlan) -> str:
    """The worked example for a release.

    The demo runs the project on one of its own evaluation cases and
    prints the input, the verdict, and the reasoning. It uses the
    deterministic path, so it needs no model and no network, and it
    shows the behaviour a reader can rely on.
    """
    demo_case = DEMO_CASES.get(plan.project, "")
    sources = [plan.project] + list(plan.borrowed_fixtures)
    provider_note = ""
    provider_helper = "    return None\n"
    provider_import = ""
    if plan.needs_model:
        provider_import = "from .core.providers import FakeProvider\n"
        provider_note = (
            "    # This project is a model judge, so the demo uses the\n"
            "    # deterministic fake provider. Point it at a real endpoint\n"
            "    # to see the real thing.\n"
        )
        provider_helper = (
            "    return FakeProvider(responses={\n"
            '        "naive_binary": "no",\n'
            '        "naive_verdict": "not enough information",\n'
            "    })\n"
        )
    return (
        '"""A worked example for the %(package)s project.\n'
        "\n"
        "The demo runs the project on one of its own evaluation cases\n"
        "and prints the input, the verdict, and the reasoning. It uses the\n"
        "deterministic path, so it needs no model and no network access.\n"
        "\n"
        "The expected label is shown for comparison. Every label in this\n"
        "package is model drafted and on the development split, so the\n"
        "demo shows agreement on a small tuned case set, not accuracy on\n"
        "real inputs.\n"
        '"""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "import json\n"
        "\n"
        "from .core.fixtures import load_cases\n"
        "%(provider_import)s"
        "from .%(module)s import build_runner\n"
        "\n"
        "DEMO_CASE = %(demo_case)r\n"
        "FIXTURE_SOURCES = %(sources)r\n"
        "\n"
        "\n"
        "def _provider():\n"
        '    """The provider the demo runs with."""\n'
        "%(provider_helper)s"
        "\n"
        "\n"
        "def _all_cases():\n"
        '    """Every case the package ships, across its fixture files."""\n'
        "    cases = []\n"
        '    for source in FIXTURE_SOURCES:\n'
        '        for name in ("cases.json", "test_cases.json", "hard_cases.json",\n'
        '                     "runtime_cases.json"):\n'
        "            cases.extend(load_cases(source, name))\n"
        "    return cases\n"
        "\n"
        "\n"
        "def _find(case_id: str):\n"
        '    """The named case, or the first case if the name is unknown."""\n'
        "    cases = _all_cases()\n"
        "    for case in cases:\n"
        "        if case.case_id == case_id:\n"
        "            return case\n"
        "    return cases[0] if cases else None\n"
        "\n"
        "\n"
        "def run_demo() -> None:\n"
        '    """Run the demo and print the result."""\n'
        "    case = _find(DEMO_CASE)\n"
        "    if case is None:\n"
        '        print("no cases are packaged with this project")\n'
        "        return\n"
        "\n"
        '    print("case:      ", case.case_id)\n'
        '    print("split:     ", case.split.value)\n'
        '    print("label:     ", case.label_origin.value)\n'
        '    print()\n'
        '    print("input:")\n'
        "    for line in json.dumps(case.input, indent=2).splitlines():\n"
        '        print("   ", line)\n'
        '    print()\n'
        "\n"
        "    result = build_runner(provider=_provider()).evaluate_case(case)\n"
        "%(provider_note)s"
        '    print("verdict:   ", result.predicted.get("verdict"))\n'
        '    print("expected:  ", case.expected.get("verdict"))\n'
        '    print()\n'
        "    for line in (result.notes or \"\").splitlines():\n"
        '        print("   ", line)\n'
    ) % {
        "package": plan.package,
        "module": module_name(plan.project),
        "project": plan.project,
        "demo_case": demo_case,
        "sources": sources,
        "provider_note": provider_note,
        "provider_helper": provider_helper,
        "provider_import": provider_import,
    }


def _first_line(project: str) -> str:
    """The one line description of a project, taken from its module."""
    path = _lab_root() / "ai_reliability" / "projects" / (project + ".py")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "A tool from the AI reliability lab."
    for index, line in enumerate(lines):
        if line.strip() == '"""' and index + 1 < len(lines):
            return lines[index + 1].strip()
    return "A tool from the AI reliability lab."


def _release_summary(plan: ReleasePlan) -> str:
    """The one line description used in the package metadata."""
    from .release_content import CONTENT

    content = CONTENT.get(plan.project)
    if content is None:
        return _first_line(plan.project)
    return content.summary


def _init_module(plan: ReleasePlan) -> str:
    """The package init file for a release."""
    return (
        '"""%(package)s.\n'
        "\n"
        "%(summary)s\n"
        '"""\n'
        "\n"
        '__version__ = "%(version)s"\n'
        "\n"
        "from .%(module)s import *  # noqa: F401,F403\n"
        "from . import %(module)s  # noqa: F401\n"
    ) % {
        "package": plan.package,
        "summary": _release_summary(plan),
        "module": module_name(plan.project),
        "version": RELEASE_VERSION,
    }


def _fallback_test(plan: ReleasePlan) -> str:
    """A minimal test file for a project that has none of its own yet."""
    package = package_dir_name(plan.project)
    sources = [plan.project] + list(plan.borrowed_fixtures)
    provider_helper = "    return None\n"
    provider_import = ""
    if plan.needs_model:
        provider_import = (
            "from " + package + ".core.providers import FakeProvider\n"
        )
        provider_helper = (
            "    return FakeProvider(responses={\n"
            '        "naive_binary": "no",\n'
            '        "naive_verdict": "not enough information",\n'
            "    })\n"
        )
    return (
        '"""Tests for the %(package)s project.\n'
        "\n"
        "These check the properties that matter for a reliability tool: it\n"
        "must not accuse without evidence, and it must abstain when the\n"
        "evidence does not settle the question.\n"
        '"""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "from %(package)s.core.fixtures import load_cases\n"
        "%(provider_import)s"
        "from %(package)s.%(module)s import build_runner\n"
        "\n"
        "FIXTURE_SOURCES = %(sources)r\n"
        "\n"
        "def _provider():\n"
        '    """The provider the tests run with."""\n'
        "%(provider_helper)s"
        "\n"
        "\n"
        "def _all_cases():\n"
        '    """Every case the package ships."""\n'
        "    cases = []\n"
        "    for source in FIXTURE_SOURCES:\n"
        "        cases.extend(load_cases(source))\n"
        "    return cases\n"
        "\n"
        "\n"
        "def test_the_project_ships_cases():\n"
        '    """The package must ship evaluation cases of its own."""\n'
        "    cases = _all_cases()\n"
        "    assert cases, \"no evaluation cases are packaged\"\n"
        "\n"
        "\n"
        "def test_every_case_runs_and_is_scored():\n"
        '    """Every packaged case must run and produce a result."""\n'
        "    cases = _all_cases()\n"
        "    runner = build_runner(provider=_provider())\n"
        "    for case in cases:\n"
        "        result = runner.evaluate_case(case)\n"
        "        assert result.predicted, case.case_id\n"
        "        assert result.correct is not None, case.case_id\n"
    ) % {
        "package": package,
        "module": module_name(plan.project),
        "project": plan.project,
        "sources": sources,
        "provider_helper": provider_helper,
        "provider_import": provider_import,
    }


def _results_section(plan: ReleasePlan) -> str:
    """The measured results for a project, read from the run files.

    The numbers are read rather than written so they cannot go stale.
    When there is no run file the section says so, because a release
    that claims results it cannot show is worse than one that admits it
    has none.
    """
    runs_dir = _lab_root().parent / "runs"
    candidates = [
        runs_dir / ("%s_model.json" % plan.project),
        runs_dir / ("%s_baseline.json" % plan.project),
        runs_dir / ("%s.json" % plan.project),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        splits = data.get("splits") or {}
        origins = data.get("label_origins") or {}
        lines = [
            "| Metric | Value |",
            "|--------|-------|",
            "| Cases | %s |" % data.get("total_cases", "?"),
            "| Scored | %s |" % data.get("scored_cases", "?"),
            "| Accepted by the fixture scorer | %s |" % data.get("correct_cases", "?"),
            "| Splits | %s |" % (
                ", ".join("%s %s" % (k, v) for k, v in splits.items()) or "?"
            ),
            "| Label origins | %s |" % (
                ", ".join("%s %s" % (k, v) for k, v in origins.items()) or "?"
            ),
        ]
        for metric in data.get("metrics", []):
            lines.append(
                "| %s | %s of %s |" % (
                    metric.get("name", "?"),
                    metric.get("numerator", "?"),
                    metric.get("denominator", "?"),
                )
            )
        return "\n".join(lines)
    return (
        "No run file ships with this package. Run the `eval` command to "
        "produce one, and read the caveats below before quoting any number "
        "it produces."
    )


def _readme(plan: ReleasePlan) -> str:
    """The README for a release."""
    from .release_content import CONTENT

    content = CONTENT.get(plan.project)
    if content is None:
        content = CONTENT["claims"]

    stages = "\n".join("%d. %s" % (i, s) for i, s in enumerate(content.stages, 1))
    rules = content.bullets(content.rules)
    limits = content.bullets(content.limits)
    results = _results_section(plan)
    module = module_name(plan.project)

    return (
        "# %(package)s\n"
        "\n"
        "%(summary)s\n"
        "\n"
        "## Why this exists\n"
        "\n"
        "%(why)s\n"
        "\n"
        "## Install\n"
        "\n"
        "Python 3.11 or newer. There are no runtime dependencies.\n"
        "\n"
        "```bash\n"
        "pip install .\n"
        "```\n"
        "\n"
        "Or run it straight from the source tree:\n"
        "\n"
        "```bash\n"
        "export PYTHONPATH=src\n"
        "```\n"
        "\n"
        "## Use\n"
        "\n"
        "```bash\n"
        "%(package)s demo\n"
        "```\n"
        "\n"
        "The demo runs the project on one of its own evaluation cases and\n"
        "prints the input, the verdict, and the reasoning. It uses the\n"
        "deterministic path, so it needs no model and no network access.\n"
        "\n"
        "To run the project's own evaluation cases:\n"
        "\n"
        "```bash\n"
        "%(package)s eval\n"
        "```\n"
        "\n"
        "## How it works\n"
        "\n"
        "%(stages)s\n"
        "\n"
        "## The rules that keep it honest\n"
        "\n"
        "These are the rules that stop the tool producing false\n"
        "accusations, which is the failure mode that matters most for a\n"
        "reliability tool.\n"
        "\n"
        "%(rules)s\n"
        "\n"
        "## What the results mean\n"
        "\n"
        "%(results)s\n"
        "\n"
        "Every label in this package is model drafted and on the development\n"
        "split. The numbers are agreement counts over a small tuned case\n"
        "set, not accuracy estimates, and no independent human review has\n"
        "been done. Read them as a check that the code does what it says,\n"
        "not as a claim about how well it works in the world.\n"
        "\n"
        "## Known limitations\n"
        "\n"
        "%(limits)s\n"
        "\n"
        "## Layout\n"
        "\n"
        "```\n"
        "%(package)s/\n"
        "  README.md\n"
        "  CHANGELOG.md\n"
        "  pyproject.toml\n"
        "  src/%(module)s/\n"
        "    %(module)s.py    the project itself\n"
        "    cli.py           the demo and eval commands\n"
        "    demo.py          the worked example\n"
        "    core/            the shared types, retrieval, and providers\n"
        "  fixtures/%(project)s/   the evaluation cases\n"
        "  tests/            the project's own test suite\n"
        "```\n"
        "\n"
        "## Licence\n"
        "\n"
        "MIT. All fixtures are synthetic.\n"
    ) % {
        "package": plan.package,
        "summary": content.summary,
        "question": content.question,
        "why": content.why,
        "stages": stages,
        "rules": rules,
        "limits": limits,
        "results": results,
        "module": module,
        "project": plan.project,
    }


def build_release(
    project: str,
    out_dir: Path,
    lab_root: Path | None = None,
) -> dict[str, Any]:
    """Write the standalone package for a project.

    Returns a report describing what was written and what was checked.
    """
    root = lab_root or _lab_root()
    plan = plan_release(project, root)
    if not plan.releasable:
        raise ValueError("%s cannot be released: %s" % (project, plan.reason))

    target = Path(out_dir) / plan.project
    source_pkg = target / "src" / package_dir_name(plan.project)
    core_dir = root / "ai_reliability" / "core"
    projects_dir = root / "ai_reliability" / "projects"
    fixtures_dir = root.parent / "fixtures" / plan.project

    # The project module, with its core imports rewritten.
    project_text = (projects_dir / (project + ".py")).read_text(encoding="utf-8")
    _write(source_pkg / (module_name(project) + ".py"), _rewrite_imports(project_text))

    # The core modules it needs.
    core_target = source_pkg / "core"
    core_target.mkdir(parents=True, exist_ok=True)
    _write(
        core_target / "__init__.py",
        '"""Shared core: the types, retrieval, and provider plumbing."""\n',
    )
    # The evaluation layer is always present, because the release
    # provides an `eval` command.
    for name in sorted(set(plan.core_modules) | {"evaluation"}):
        shutil.copyfile(core_dir / (name + ".py"), core_target / (name + ".py"))

    # The package files.
    _write(source_pkg / "__init__.py", _init_module(plan))
    _write(source_pkg / "cli.py", _cli_source(plan))
    _write(source_pkg / "demo.py", _demo_source(plan))

    # The fixtures.
    if fixtures_dir.is_dir():
        shutil.copytree(
            fixtures_dir,
            target / "fixtures" / plan.project,
            dirs_exist_ok=True,
        )
        # The fixtures go inside the package as well, so that an
        # installed wheel carries its own cases. A package that
        # installs cleanly and then cannot find its fixtures is not
        # finished.
        shutil.copytree(
            fixtures_dir,
            source_pkg / "fixtures" / plan.project,
            dirs_exist_ok=True,
        )
    for borrowed in plan.borrowed_fixtures:
        source = root.parent / "fixtures" / borrowed
        if source.is_dir():
            shutil.copytree(
                source,
                target / "fixtures" / borrowed,
                dirs_exist_ok=True,
            )
            shutil.copytree(
                source,
                source_pkg / "fixtures" / borrowed,
                dirs_exist_ok=True,
            )

    # The distribution metadata.
    _write(target / "pyproject.toml", _pyproject(plan, _release_summary(plan)))
    _write(target / "MANIFEST.in", _manifest())
    _write(target / "README.md", _readme(plan))
    _write(target / "CHANGELOG.md", _changelog(plan))
    licence = root.parent / "LICENSE"
    if licence.exists():
        shutil.copyfile(licence, target / "LICENSE")

    # The project's own test file.
    tests_source = root.parent / "tests" / "projects" / ("test_%s.py" % project)
    if tests_source.exists():
        _write(
            target / "tests" / ("test_%s.py" % project),
            _rewrite_test_imports(
                tests_source.read_text(encoding="utf-8"), plan
            ),
        )
    else:
        _write(target / "tests" / ("test_%s.py" % project), _fallback_test(plan))

    report = verify_release(target)
    report["project"] = project
    report["package"] = plan.package
    report["core_modules"] = plan.core_modules
    report["fixture_files"] = plan.fixture_files
    return report


def verify_release(release_dir: Path) -> dict[str, Any]:
    """Check that a release directory stands on its own.

    The rule is that a release must not import the lab. A release
    that reached back into the lab for one helper would not be
    independent, and the whole point of extracting a project is that
    it stands alone.
    """
    release_dir = Path(release_dir)
    problems: list[str] = []
    checked = 0

    for path in sorted(release_dir.rglob("*.py")):
        checked += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            problems.append("%s does not parse: %s" % (path.name, exc))
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names = [node.module]
                names = names + [
                    "%s.%s" % (node.module or "", alias.name)
                    for alias in node.names
                ]
            for name in names:
                if name.startswith("ai_reliability"):
                    problems.append(
                        "%s still imports %s" % (path.name, name)
                    )
            if isinstance(node, ast.ImportFrom) and node.level == 2:
                problems.append(
                    "%s still uses a two level relative import" % path.name
                )

    required = ["pyproject.toml", "README.md", "CHANGELOG.md", "LICENSE"]
    for name in required:
        if not (release_dir / name).exists():
            problems.append("missing %s" % name)

    return {
        "directory": str(release_dir),
        "python_files_checked": checked,
        "independent": not problems,
        "problems": problems,
    }
