"""Command-line interface for the claim auditor.

Every command prints JSON by default so results can be piped into
other tools.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .evaluation import (
    ClaimRunner,
    Split,
    load_cases,
    run_evaluation,
    save_run,
)
from .providers import ENV_ENDPOINT, ENV_MODEL, ProviderError, OllamaProvider


def _provider(args: argparse.Namespace):
    if getattr(args, "fake", False):
        from .providers import FakeProvider

        return FakeProvider()
    if getattr(args, "offline", False):
        return None
    return OllamaProvider(model_id=getattr(args, "model", None), url=getattr(args, "endpoint", None))


def _add_model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help=f"model id (default: ${ENV_MODEL})")
    parser.add_argument("--endpoint", default=None, help=f"Ollama base URL (default: ${ENV_ENDPOINT})")
    parser.add_argument("--offline", action="store_true", help="never contact a model")
    parser.add_argument("--fake", action="store_true", help="use the deterministic fake provider")


def cmd_audit(args: argparse.Namespace) -> int:
    """Audit one answer against supplied sources."""
    from .auditor import SourceDocument, audit_claims

    answer = args.answer
    sources = []
    for index, item in enumerate(args.sources or []):
        identifier, _, body = item.partition("=")
        if not body:
            # No "=" means the whole argument is the text.
            identifier, body = f"source_{index + 1}", item
        sources.append(
            SourceDocument(identifier=identifier.strip() or f"source_{index + 1}", text=body)
        )
    provider = _provider(args)
    report = audit_claims(answer, sources, provider=provider)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(report.render())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nreport written to {args.out}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Run the evaluation suite over a fixture file."""
    if Path(args.cases).is_dir():
        from .evaluation import load_all_cases

        cases = load_all_cases(args.cases)
    else:
        cases = load_cases(args.cases)
    if args.split:
        wanted = Split(args.split)
        cases = [case for case in cases if case.split is wanted]
    if not cases:
        print("no cases matched", file=sys.stderr)
        return 1
    provider = _provider(args)
    runner = ClaimRunner(provider=provider)
    report = run_evaluation(
        cases, runner, method=args.method, model_id=getattr(provider, "model_id", None)
    )
    if args.out:
        save_run(report, args.out)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        print(report.render())
        if args.out:
            print(f"\nrun written to {args.out}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check the model endpoint."""
    provider = _provider(args)
    payload: dict[str, Any] = {
        "provider": type(provider).__name__,
        "model_id": getattr(provider, "model_id", None),
        "endpoint": os.environ.get(ENV_ENDPOINT, "http://127.0.0.1:11434"),
    }
    if provider is not None:
        try:
            import urllib.request

            with urllib.request.urlopen(provider.url + "/api/tags", timeout=5) as response:
                response.read()
            payload["reachable"] = True
        except Exception as exc:
            payload["reachable"] = False
            payload["detail"] = str(exc)
    else:
        payload["reachable"] = None
        payload["detail"] = "offline mode"
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0 if payload.get("reachable") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claim-auditor",
        description="Check whether an AI answer is supported by the sources it was given.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="audit one answer against sources")
    audit.add_argument("answer", help="the answer text to check")
    audit.add_argument(
        "--source",
        action="append",
        dest="sources",
        metavar="IDENTIFIER=TEXT",
        help="a source to check against, as IDENTIFIER=TEXT; repeat for several",
    )
    audit.add_argument("--out", default=None, help="write the JSON report here")
    audit.add_argument("--json", action="store_true")
    _add_model_flags(audit)
    audit.set_defaults(func=cmd_audit)

    evaluate = sub.add_parser("eval", help="run the evaluation suite")
    evaluate.add_argument("--cases", default="fixtures/cases.json")
    evaluate.add_argument("--split", choices=[s.value for s in Split], default=None)
    evaluate.add_argument("--method", default="hybrid")
    evaluate.add_argument("--out", default=None)
    evaluate.add_argument("--json", action="store_true")
    _add_model_flags(evaluate)
    evaluate.set_defaults(func=cmd_eval)

    doctor = sub.add_parser("doctor", help="check the model endpoint")
    _add_model_flags(doctor)
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130
    except (ValueError, ProviderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
