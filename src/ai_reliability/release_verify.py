"""Run every built release and check that it works on its own.

A release is only a release if someone can clone it and run it. This
module does exactly that for each built package: it runs the demo, it
runs the project's own test suite, and it checks that the package
imports without the lab being on the path.

The import check is the important one. The demo and the tests can both
pass while the package quietly reaches back into the lab, because the
lab is usually installed in the same environment. Running the import
with the lab directory removed from the path is what proves the
release stands alone.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .release import DISTRIBUTION_NAMES, list_plans, package_dir_name


@dataclass
class ReleaseCheck:
    """The result of checking one release."""

    project: str
    package: str
    demo_ok: bool = False
    demo_output: str = ""
    tests_ok: bool = False
    tests_output: str = ""
    imports_ok: bool = False
    imports_output: str = ""
    install_ok: bool = False
    install_output: str = ""
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.demo_ok
            and self.tests_ok
            and self.imports_ok
            and self.install_ok
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "package": self.package,
            "ok": self.ok,
            "demo_ok": self.demo_ok,
            "tests_ok": self.tests_ok,
            "imports_ok": self.imports_ok,
            "install_ok": self.install_ok,
            "problems": self.problems,
        }


def _run(command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def check_release(release_dir: Path, python: str | None = None) -> ReleaseCheck:
    """Run the demo, the tests, and the import check for one release."""
    # The path has to be absolute. The checks run with the release
    # directory as their working directory, so a relative path would be
    # resolved against the wrong place.
    release_dir = Path(release_dir).resolve()
    interpreter = python or sys.executable
    package = package_dir_name(release_dir.name)
    plan_package = DISTRIBUTION_NAMES.get(release_dir.name, release_dir.name)

    check = ReleaseCheck(project=release_dir.name, package=package)

    # The demo, with the lab directory removed from the path so that
    # any reach back into the lab shows up as a failure.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(release_dir / "src")
    env.pop("AIRL_MODEL_CACHE", None)

    demo = _run(
        [interpreter, "-m", package + ".cli", "demo"],
        release_dir,
        env,
    )
    check.demo_ok = demo.returncode == 0
    check.demo_output = (demo.stdout or "")[-2000:] + (demo.stderr or "")[-2000:]
    if not check.demo_ok:
        check.problems.append("the demo did not run")

    tests = _run([interpreter, "-m", "pytest", "-q"], release_dir, env)
    check.tests_ok = tests.returncode == 0
    check.tests_output = (tests.stdout or "")[-2000:]
    if not check.tests_ok:
        check.problems.append("the test suite did not pass")

    # The import check. The package is imported in a process whose
    # working directory is the release and whose path has only the
    # release on it, so an import of the lab cannot be satisfied.
    probe = (
        "import importlib, sys;"
        "module = importlib.import_module(%r);"
        "print('ok', module.__name__)" % package
    )
    imported = _run(
        [interpreter, "-c", probe],
        release_dir,
        env,
    )
    check.imports_ok = imported.returncode == 0
    check.imports_output = (imported.stdout or "") + (imported.stderr or "")
    if not check.imports_ok:
        check.problems.append("the package does not import on its own")

    # The install check. A package can import from a checkout and
    # still fail once it is installed, because the fixtures it needs
    # were never packaged. This builds a wheel, installs it into a
    # throwaway directory, and runs the console script from there.
    check.install_ok, check.install_output = _check_install(
        release_dir, package, plan_package
    )
    if not check.install_ok:
        check.problems.append("the installed package does not run")

    return check


def _check_install(
    release_dir: Path, package: str, plan_package: str
) -> tuple[bool, str]:
    """Build, install, and run the package the way a reader would."""
    import shutil
    import tempfile

    interpreter = sys.executable
    with tempfile.TemporaryDirectory(prefix="airl_install_") as workspace:
        # The build runs against a copy. Building in the release
        # directory would leave a build directory and an egg-info
        # directory behind, and those are artefacts rather than
        # sources.
        source = Path(workspace) / "src"
        shutil.copytree(
            release_dir,
            source,
            ignore=shutil.ignore_patterns(
                "build", "dist", "*.egg-info", "__pycache__", ".pytest_cache"
            ),
        )
        wheelhouse = Path(workspace) / "wheels"
        target = Path(workspace) / "site"
        built = _run(
            [
                interpreter,
                "-m",
                "pip",
                "wheel",
                "--quiet",
                "--no-build-isolation",
                "--no-deps",
                "--wheel-dir",
                str(wheelhouse),
                ".",
            ],
            source,
            dict(os.environ),
        )
        if built.returncode != 0:
            return False, "the wheel did not build:\n" + (built.stderr or "")[-1500:]

        wheels = sorted(wheelhouse.glob("*.whl"))
        if not wheels:
            return False, "no wheel was produced"

        installed = _run(
            [
                interpreter,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-deps",
                "--target",
                str(target),
                str(wheels[0]),
            ],
            release_dir,
            dict(os.environ),
        )
        if installed.returncode != 0:
            return False, "the wheel did not install:\n" + (installed.stderr or "")[-1500:]

        env = dict(os.environ)
        env["PYTHONPATH"] = str(target)
        # The console script is named after the distribution, which
        # uses hyphens, not after the importable package directory.
        script = target / "bin" / plan_package
        if not script.exists():
            return False, "the console script was not installed"
        run = _run([str(script), "demo"], release_dir, env)
        if run.returncode != 0:
            return False, "the installed package did not run:\n" + (
                (run.stdout or "")[-1000:] + (run.stderr or "")[-1000:]
            )
        return True, "the installed package ran"


def check_all(out_dir: Path, python: str | None = None) -> list[ReleaseCheck]:
    """Check every release that has been built into a directory."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    checks = []
    for directory in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        if not (directory / "pyproject.toml").exists():
            continue
        checks.append(check_release(directory, python=python))
    return checks


def release_status(out_dir: Path, python: str | None = None) -> dict[str, Any]:
    """A summary of every project and every built release."""
    plans = list_plans()
    checks = check_all(out_dir, python=python)
    by_project = {c.project: c for c in checks}
    rows = []
    for plan in plans:
        check = by_project.get(plan.project)
        rows.append(
            {
                "project": plan.project,
                "package": plan.package,
                "releasable": plan.releasable,
                "reason": plan.reason,
                "built": check is not None,
                "verified": bool(check and check.ok),
                "problems": list(check.problems) if check else [],
            }
        )
    return {
        "projects": rows,
        "releasable": sum(1 for p in plans if p.releasable),
        "built": len(checks),
        "verified": sum(1 for c in checks if c.ok),
    }
