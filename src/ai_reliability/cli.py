"""Command-line interface for the AI reliability lab.

Every project is reachable from the CLI so the lab can be demonstrated
without a web UI:

    airl audit receipt   ...
    airl audit deliverable ...
    airl audit claims    ...
    airl audit citations ...
    airl audit omissions ...
    airl audit eval      ...

All commands print JSON by default so results can be piped into other
tools, and accept ``--pretty`` for a human-readable rendering.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .core.evaluation import load_run
from .core.providers import ENV_ENDPOINT, ENV_MODEL, ProviderError, build_provider, provider_available
from .core.types import Split

PROJECT_NAMES = [
    "omission",
    "claims",
    "citations",
    "deliverables",
    "context_receipts",
    "eval_regression",
    "numerical",
    "instructions",
    "code_review",
    "authorization",
    "source_versions",
    "study_coverage",
    "retrieval_eval",
    "baselines",
]


def _emit(payload: Any, pretty: bool) -> None:
    if pretty and isinstance(payload, dict) and "render" in payload:
        print(payload["render"])
        return
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _provider(args: argparse.Namespace):
    if getattr(args, "fake", False):
        from .core.providers import FakeProvider

        return FakeProvider()
    return build_provider(
        model_id=getattr(args, "model", None),
        url=getattr(args, "endpoint", None),
        offline=getattr(args, "offline", False),
    )


def _add_model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=None,
        help=f"model id to use (default: ${ENV_MODEL} or the built-in default)",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help=f"Ollama-compatible base URL (default: ${ENV_ENDPOINT} or http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never contact a model; use the deterministic path only",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="use the deterministic fake provider instead of a real model",
    )


# --------------------------------------------------------------------------
# receipt
# --------------------------------------------------------------------------


def cmd_receipt(args: argparse.Namespace) -> int:
    from .projects.context_receipts import build_receipt, render_receipt

    receipt = build_receipt(args.files, created_at=args.created_at)
    if args.json:
        print(receipt.to_json())
    else:
        print(render_receipt(receipt))
        if args.out:
            Path(args.out).write_text(receipt.to_json() + "\n", encoding="utf-8")
            print(f"\nreceipt written to {args.out}")
    return 0



def cmd_receipt_diff(args: argparse.Namespace) -> int:
    from .projects.context_receipts import build_receipt, diff_receipts, render_diff

    before = build_receipt(args.before, created_at=None)
    after = build_receipt(args.after, created_at=args.created_at)
    diff = diff_receipts(before, after)
    if args.json:
        print(json.dumps(diff, indent=2, ensure_ascii=False))
    else:
        print(render_diff(diff))
    return 0

def cmd_deliverable(args: argparse.Namespace) -> int:
    from .projects.deliverables import Requirement, check_deliverable

    request = _read_text(args.request)
    deliverable = _read_text(args.deliverable) if args.deliverable else ""
    requirements = None
    if args.requirements:
        payload = json.loads(Path(args.requirements).read_text(encoding="utf-8"))
        entries = payload.get("requirements", []) if isinstance(payload, dict) else payload
        requirements = [Requirement.from_dict(entry) for entry in entries]
    report = check_deliverable(request, deliverable, requirements=requirements)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(report.render())
    return 0


def cmd_requirements(args: argparse.Namespace) -> int:
    """Extract the requirement list so a user can review and edit it.

    This is the first half of the review-and-recheck flow. The output is
    written in exactly the format ``--requirements`` reads back, so the
    round trip is: extract, edit, recheck.
    """
    from .projects.deliverables import extract_requirements

    request = _read_text(args.request)
    requirements = extract_requirements(request)
    payload = {
        "request": request,
        "instructions": (
            "Edit any field, then recheck with:"
            " audit deliverable <request> --deliverable <file>"
            " --requirements <this file>"
        ),
        "requirements": [r.to_dict() for r in requirements],
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"requirements written to {args.out} ({len(requirements)} requirement(s))")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _read_text(path: str) -> str:
    """Read a text input: a literal string, or a path to a file."""
    candidate = Path(path)
    if candidate.exists() and candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return path
# eval runner
# --------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    from .projects.registry import get_project, run_project_evaluation

    provider = _provider(args)
    split = Split(args.split) if args.split else None
    if args.fixture:
        from .core.evaluation import run_evaluation
        from .core.fixtures import load_cases

        entry = get_project(args.project)
        cases = load_cases(args.project, args.fixture)
        if split:
            cases = [case for case in cases if case.split is split]
        if not cases:
            raise ValueError("the selected fixture contains no cases")
        runner = (
            entry._module().build_runner(provider, kind=args.baseline_kind)
            if args.project == "baselines" else entry.build_runner(provider)
        )
        report = run_evaluation(
            args.project, cases, runner,
            method=args.method or entry.method,
            model_id=getattr(provider, "model_id", None),
        )
    else:
        report = run_project_evaluation(
            args.project, provider=provider, split=split,
            method=args.method, baseline_kind=args.baseline_kind,
        )
    payload = report.to_dict()
    if args.project == "retrieval_eval":
        from .projects.retrieval_eval import summarise

        payload["retrieval_summary"] = summarise(report.results)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        print(report.render())
        if args.out:
            print(f"\nrun written to {args.out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .core.evaluation import compare_runs

    left = load_run(args.left)
    right = load_run(args.right)
    result = compare_runs(left, right)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    provider = _provider(args)
    ok, detail = provider_available(provider)
    payload: dict[str, Any] = {
        "provider": type(provider).__name__,
        "model_id": getattr(provider, "model_id", None),
        "endpoint": os.environ.get(ENV_ENDPOINT, "http://127.0.0.1:11434"),
        "reachable": None if getattr(provider, "cache_only", False) else ok,
        "cache_ready": ok if getattr(provider, "cache_only", False) else None,
        "detail": detail,
    }
    if ok and hasattr(provider, "list_models") and not getattr(provider, "cache_only", False):
        payload["models"] = [
            {
                "name": m.get("name"),
                "context_length": (m.get("details") or {}).get("context_length"),
                "size": (m.get("details") or {}).get("parameter_size"),
            }
            for m in provider.list_models()
        ]
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0 if ok else 1


def cmd_record(args: argparse.Namespace) -> int:
    """Record model responses for a set of prompts into a cache file.

    The prompts are built by the project that needs them, then each one
    is sent to the endpoint. This exists so a model-assisted evaluation
    can be replayed later without an endpoint, and so a reviewer can see
    exactly what the model was asked and what it said.
    """
    from .core.providers import ProviderError, cache_key

    import importlib

    module = importlib.import_module(f"ai_reliability.projects.{args.project}")
    builder = getattr(module, "build_prompts")

    prompts: list[tuple[str, str]] = list(builder(args.fixture))
    provider = build_provider(model_id=args.model, url=args.endpoint)
    responses: dict[str, str] = {}
    failures: list[str] = []
    for purpose, prompt in prompts:
        # A cloud relay can drop a connection mid-run. A retry keeps one
        # flaky call from discarding an otherwise complete recording, and
        # the retry is recorded so the run is not mistaken for a clean
        # first pass.
        result = None
        last_error = ""
        for attempt in range(1, args.retries + 1):
            try:
                result = provider.generate(
                    prompt, purpose=purpose, prompt_version="record-v1",
                    json_mode=purpose not in {"naive_binary", "naive_verdict"},
                    timeout=args.timeout,
                )
                break
            except ProviderError as exc:
                last_error = str(exc)
                print(
                    f"call failed (attempt {attempt}/{args.retries}): {exc}",
                    file=sys.stderr,
                )
                if attempt < args.retries:
                    time.sleep(args.retry_delay)
        if result is None:
            failures.append(f"{purpose}: {last_error}")
            continue
        key = cache_key(purpose, prompt)
        responses[key] = result.text
        print(f"recorded {key[:24]} ({len(result.text)} chars)")

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8")).get("responses", {})
        except Exception:
            existing = {}
    existing.update(responses)
    target.write_text(
        json.dumps(
            {
                "model": provider.model_id,
                "responses": existing,
                "failed_prompts": failures,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print("cache written to " + str(target) + " (" + str(len(existing)) + " responses)")
    if failures:
        print(
            "%d prompt(s) failed after %d attempt(s) each and are NOT in the cache"
            % (len(failures), args.retries),
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_prompts(args: argparse.Namespace) -> int:
    """Print the prompts a project would send, one JSON object per line.

    Used with the record command so a caller can inspect or replay the
    exact prompts without running the auditor.
    """
    import importlib

    module = importlib.import_module(f"ai_reliability.projects.{args.project}")
    builder = getattr(module, "build_prompts")

    for purpose, prompt in builder(args.fixture):
        print(json.dumps({"purpose": purpose, "prompt": prompt}, ensure_ascii=False))
    return 0


def cmd_projects(args: argparse.Namespace) -> int:
    from .projects.registry import project_status

    print(json.dumps(project_status(), indent=2, ensure_ascii=False, default=str))
    return 0


def _review_cases():
    """Collect primary, optional, and separate hard-case fixtures."""
    from .projects.eval_regression import COVERED_PROJECTS, build_cases
    from .core.fixtures import load_cases

    cases = build_cases([*COVERED_PROJECTS, "source_versions", "study_coverage"])
    for project in ("claims", "citations", "deliverables", "code_review"):
        cases.extend(load_cases(project, "hard_cases.json"))
    return cases


def cmd_review_export(args: argparse.Namespace) -> int:
    """Export blinded cases for independent human review."""
    cases = _review_cases()
    packet = {
        "instructions": (
            "For each case, inspect only the input and write your own expected outcome "
            "without consulting fixtures or run predictions. Set reviewed_expected to "
            "an object such as {\"verdict\": \"supported\"}; add other expected fields "
            "only if you independently checked them. Leave null if unreviewed. "
            "Use reviewer_notes for uncertainty."
        ),
        "cases": [
            {"case_id": case.case_id, "project": case.project, "input": case.input,
             "reviewed_expected": None, "reviewer_notes": ""}
            for case in cases
        ],
    }
    Path(args.out).write_text(json.dumps(packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"exported {len(cases)} blinded cases to {args.out}")
    return 0
