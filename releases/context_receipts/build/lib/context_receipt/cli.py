"""Command line interface for the context-receipt project.

Two commands:

    context-receipt demo    run the project on a worked example
    context-receipt eval    run the project's evaluation cases

Both work with no model and no network access. The demo uses the
deterministic path, so it shows the behaviour a reader can rely
on without a model endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core.evaluation import run_evaluation
from .core.fixtures import load_cases
from .context_receipts import build_cases, build_runner

FIXTURE_SOURCES = ['context_receipts']


def _fixtures_dir() -> Path:
    """Where the packaged fixtures live."""
    here = Path(__file__).resolve().parent.parent.parent
    return here / "fixtures" / "context_receipts"


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the project on a worked example."""
    from .demo import run_demo

    run_demo()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Run the project's own evaluation cases."""
    cases = []
    for source in FIXTURE_SOURCES:
        cases.extend(load_cases(source, args.fixture))
    if not cases:
        print("no cases found in", _fixtures_dir(), file=sys.stderr)
        return 1
    runner = build_runner(provider=None)
    report = run_evaluation(
        "context_receipts", cases, runner, method="deterministic"
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(report.render())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="context-receipt",
        description="",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the worked example")
    demo.set_defaults(func=cmd_demo)

    evaluate = sub.add_parser("eval", help="run the evaluation cases")
    evaluate.add_argument("--fixture", default="cases.json")
    evaluate.add_argument("--json", action="store_true")
    evaluate.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
