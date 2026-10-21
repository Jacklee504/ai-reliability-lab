"""Model provider adapters.

One adapter talks to Ollama over its local HTTP API; one is a
deterministic fake used by tests and by the no-model-access fallback.

Rules enforced here:

* No API key is required or read. The endpoint is configurable by
  environment variable or CLI flag so a hosted provider can replace
  MiniMax later without touching project code.
* Every provider call records model id, prompt version, and latency.
  Project parsers record response parse failures in evaluation results.
* ``generate`` is synchronous. Callers should issue one request at a time
  unless their endpoint's concurrency has been verified.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .types import sha256_text

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "minimax-m3.1-flash:cloud"

ENV_ENDPOINT = "AIRL_OLLAMA_URL"
ENV_MODEL = "AIRL_MODEL"

#: Optional path to a JSON file of cached model responses.
#:
#: Keys are ``"<purpose>::<hash of the prompt>"``; values are the raw
#: response text. When the cache is present, a call is served from it
#: instead of the network. This makes a model-assisted evaluation
#: reproducible and runnable with no endpoint at all.
ENV_CACHE = "AIRL_MODEL_CACHE"


class ProviderError(RuntimeError):
    """Raised when the model endpoint is unreachable or returns an error."""


class ParseError(RuntimeError):
    """Raised when a model response cannot be parsed into the expected shape."""

    def __init__(self, message: str, raw: str | None = None):
        super().__init__(message)
        self.raw = raw


@dataclass
class ModelCall:
    """Provenance record for one model request."""

    call_id: str
    model_id: str
    prompt_version: str
    purpose: str
    latency_ms: float
    ok: bool
    parse_failed: bool = False
    error: str | None = None
    prompt_chars: int = 0
    response_chars: int = 0
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "purpose": self.purpose,
            "latency_ms": round(self.latency_ms, 2),
            "ok": self.ok,
            "parse_failed": self.parse_failed,
            "error": self.error,
            "prompt_chars": self.prompt_chars,
            "response_chars": self.response_chars,
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
            "extra": self.extra,
        }


@dataclass
class ModelResult:
    """Result of a model call: the text plus the call record."""

    text: str
    call: ModelCall

    def json(self) -> Any:
        """Parse the response as JSON, raising :class:`ParseError`."""
        return parse_json_block(self.text, raw=self.text)


class Provider(Protocol):
    """Minimal provider interface used by every project module."""

    model_id: str

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        purpose: str = "generic",
        prompt_version: str = "v1",
        temperature: float = 0.0,
        json_mode: bool = False,
        timeout: float = 180.0,
    ) -> ModelResult:
        ...


def parse_json_block(text: str, raw: str | None = None) -> Any:
    """Parse JSON from a model response.

    Tolerates a fenced code block and leading/trailing prose, because
    models frequently wrap JSON even when told not to. Raises
    :class:`ParseError` when nothing JSON-like is present.
    """
    candidate = text.strip()
    if not candidate:
        raise ParseError("empty model response", raw=raw)

    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Find the first balanced {...} block.
    start = candidate.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(candidate)):
            char = candidate[index]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    block = candidate[start : index + 1]
                    try:
                        return json.loads(block)
                    except json.JSONDecodeError:
                        break
        start = candidate.find("{", start + 1)

    raise ParseError("no parsable JSON in model response", raw=raw)


def cache_key(purpose: str, prompt: str) -> str:
    """Stable key for a cached model response."""
    return f"{purpose}::{sha256_text(prompt)}"


def load_cache(
    path: str | os.PathLike[str] | None = None,
    *, expected_model_id: str | None = None,
) -> dict[str, str]:
    """Load an explicitly configured cache or fail before any network call."""
    target = path or os.environ.get(ENV_CACHE)
    if not target:
        return {}
    file_path = Path(target).expanduser()
    if not file_path.exists():
        raise ProviderError(f"model cache does not exist: {file_path}")
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderError(f"cannot read model cache: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProviderError("model cache must be a JSON object")
    if expected_model_id and payload.get("model") != expected_model_id:
        raise ProviderError("model cache was recorded for a different model ID")
    responses = payload.get("responses")
    if not isinstance(responses, dict):
        raise ProviderError("model cache has no responses object")
    return {str(k): str(v) for k, v in responses.items()}
