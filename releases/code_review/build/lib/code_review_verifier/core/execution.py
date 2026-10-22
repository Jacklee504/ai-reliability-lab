"""Controlled execution of a code snapshot.

Why this exists
---------------
Some review comments are about runtime behaviour: "this will raise an
error", "this returns None", "this divides by zero". A static reading
cannot settle them, so the verifier used to answer "unverified" for
every such comment. That is honest, but it means the strongest version
of the tool is never used.

This module runs a small harness against a snapshot in a *separate
process*, with a timeout, in a temporary directory, and reports what
actually happened. The observed result is evidence; the tool never
guesses a runtime outcome.

Deliberate limits
-----------------
* Execution is **opt-in**. Nothing here runs unless a caller asks for
  it, and the evaluation fixtures that use it are a separate file.
* The harness is code the *reviewer* supplies, not code the tool
  invents. The tool never writes a test to make a comment come out a
  particular way.
* The snapshot is written into a throwaway directory and the child
  process runs in isolated mode, so it cannot see the surrounding
  environment.
* A timeout is mandatory. A comment that hangs is reported as
  "unverified", not as evidence of anything.
* A crash in the harness is reported as a crash. It is never silently
  turned into "the comment was right".

The point of this discipline is the same as everywhere else in the
lab: a wrong accusation is worse than no answer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: How long a harness may run before it is abandoned.
DEFAULT_TIMEOUT = 10.0

#: The runner script. It puts the snapshot directory on ``sys.path`` so
#: the harness can import the modules under review, and it reports the
#: outcome as a single JSON line so the parent never has to guess.
_RUNNER = '''
import json, sys, traceback

sys.path.insert(0, sys.argv[1])

outcome = {"raised": False, "exception_type": None, "exception_message": None}
try:
    namespace = {"__name__": "__harness__"}
    exec(compile(sys.argv[2], "<harness>", "exec"), namespace)
    # A harness may leave a return value in the namespace under this
    # name so the parent can observe what the code returned, not only
    # whether it raised.
    if "__airl_result__" in namespace:
        outcome["returned_none"] = namespace["__airl_result__"] is None
except BaseException as exc:
    # An ImportError, ModuleNotFoundError, or SyntaxError means the
    # *harness* is broken, not the code under review. Recording that
    # as a raised exception would let a broken harness manufacture
    # evidence against a comment, so it is reported separately and is
    # never counted as an observation.
    if isinstance(exc, (ImportError, ModuleNotFoundError, SyntaxError)):
        outcome["harness_error"] = type(exc).__name__ + ": " + str(exc)[:200]
    else:
        outcome["raised"] = True
    outcome["exception_type"] = type(exc).__name__
    outcome["exception_message"] = str(exc)[:400]
    outcome["traceback"] = traceback.format_exc()[-1200:]
sys.stdout.write("\\n__AIRL_OUTCOME__" + json.dumps(outcome))
'''


@dataclass
class ExecutionOutcome:
    """What actually happened when the harness ran.

    ``raised`` is the observed fact. ``timed_out`` and ``error`` record
    the ways the run can fail to produce an observation at all, and both
    mean the outcome is unusable.
    """

    raised: bool = False
    exception_type: str | None = None
    exception_message: str | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: str | None = None
    duration_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def observed(self) -> bool:
        """True when the run produced a usable observation."""
        return not self.timed_out and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raised": self.raised,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "returncode": self.returncode,
            "stdout": self.stdout[-2000:],
            "stderr": self.stderr[-2000:],
            "timed_out": self.timed_out,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
            "observed": self.observed,
        }


def run_harness(
    files: dict[str, str],
    harness: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    python_executable: str | None = None,
) -> ExecutionOutcome:
    """Run a harness against a snapshot and report what happened.

    ``files`` maps a relative path to the file's source. ``harness`` is
    the reviewer's code, executed with the snapshot importable.

    The child process runs in isolated mode in a temporary directory.
    Nothing outside that directory is reachable through the import
    system, and the environment is reduced to a minimum.
    """
    started = time.perf_counter()
    interpreter = python_executable or sys.executable

    with tempfile.TemporaryDirectory(prefix="airl_exec_") as workspace:
        root = Path(workspace)
        for relative, content in files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        runner = root / "_airl_runner.py"
        runner.write_text(_RUNNER, encoding="utf-8")

        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": workspace,
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "LC_ALL": "C",
        }

        try:
            completed = subprocess.run(
                [interpreter, "-I", str(runner), str(root), harness],
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecutionOutcome(
                timed_out=True,
                error="the harness did not finish within %.1fs" % timeout,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except OSError as exc:
            return ExecutionOutcome(
                error="could not start the harness: %s: %s" % (type(exc).__name__, exc),
                duration_ms=(time.perf_counter() - started) * 1000,
            )

    duration = (time.perf_counter() - started) * 1000
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    marker = "__AIRL_OUTCOME__"
    index = stdout.rfind(marker)
    if index == -1:
        return ExecutionOutcome(
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            error=(
                "the harness did not report an outcome"
                + (" and exited with a non-zero status" if completed.returncode else "")
            ),
            duration_ms=duration,
        )

    payload = stdout[index + len(marker):].strip()
    try:
        data = json.loads(payload)
    except ValueError:
        return ExecutionOutcome(
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            error="the harness outcome could not be parsed",
            duration_ms=duration,
        )

    if data.get("harness_error"):
        return ExecutionOutcome(
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            error="the harness itself failed: %s" % data["harness_error"],
            duration_ms=duration,
        )

    return ExecutionOutcome(
        raised=bool(data.get("raised")),
        exception_type=data.get("exception_type"),
        exception_message=data.get("exception_message"),
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        duration_ms=duration,
        extra={
            "traceback": data.get("traceback", ""),
            "returned_none": data.get("returned_none"),
        },
    )
