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


class OllamaProvider:
    """Adapter for a local (or remote) Ollama HTTP endpoint.

    No API key is used. The endpoint defaults to the local Ollama server
    and can be pointed anywhere via ``url`` or ``AIRL_OLLAMA_URL``.
    """

    def __init__(
        self,
        model_id: str | None = None,
        url: str | None = None,
        *,
        think: bool = False,
        default_timeout: float = 180.0,
    ):
        self.url = (url or os.environ.get(ENV_ENDPOINT) or DEFAULT_OLLAMA_URL).rstrip("/")
        self.model_id = model_id or os.environ.get(ENV_MODEL) or DEFAULT_MODEL
        self.think = think
        self.default_timeout = default_timeout
        self.cache_only = bool(os.environ.get(ENV_CACHE))
        self.cache: dict[str, str] = load_cache(expected_model_id=self.model_id)
        self.cache_hits = 0
        self.calls: list[ModelCall] = []

    def _endpoint(self, path: str) -> str:
        return f"{self.url}{path}"

    def _post(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint(path),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ProviderError(f"HTTP {exc.code} from {path}: {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise ProviderError(f"cannot reach {self._endpoint(path)}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:  # pragma: no cover - network path
            raise ProviderError(f"non-JSON response from {path}: {exc}") from exc

    def list_models(self) -> list[dict[str, Any]]:
        """Return the models the endpoint reports, or an empty list."""
        try:
            with urllib.request.urlopen(self._endpoint("/api/tags"), timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []
        return list(data.get("models", []))

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        purpose: str = "generic",
        prompt_version: str = "v1",
        temperature: float = 0.0,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> ModelResult:
        started = time.perf_counter()
        key = cache_key(purpose, prompt)
        if key in self.cache:
            self.cache_hits += 1
            text = self.cache[key]
            call = ModelCall(
                call_id=f"cache_{len(self.calls)}",
                model_id=self.model_id,
                prompt_version=prompt_version,
                purpose=purpose,
                latency_ms=(time.perf_counter() - started) * 1000,
                ok=bool(text.strip()),
                prompt_chars=len(prompt) + (len(system) if system else 0),
                response_chars=len(text),
                extra={"served_from_cache": True},
            )
            self.calls.append(call)
            return ModelResult(text=text, call=call)

        if self.cache_only:
            raise ProviderError(f"model cache miss for purpose {purpose!r}; no network call made")

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options: dict[str, Any] = {"temperature": temperature}
        if json_mode:
            options["format"] = "json"

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": options,
        }

        call = ModelCall(
            call_id=f"call_{int(started * 1000)}",
            model_id=self.model_id,
            prompt_version=prompt_version,
            purpose=purpose,
            latency_ms=0.0,
            ok=False,
            prompt_chars=len(prompt) + (len(system) if system else 0),
        )
        try:
            data = self._post("/api/chat", payload, timeout or self.default_timeout)
        except ProviderError as exc:
            call.latency_ms = (time.perf_counter() - started) * 1000
            call.error = str(exc)
            self.calls.append(call)
            raise

        message = data.get("message") or {}
        text = message.get("content") or ""
        call.latency_ms = (time.perf_counter() - started) * 1000
        call.ok = bool(text.strip())
        call.response_chars = len(text)
        call.prompt_eval_count = data.get("prompt_eval_count")
        call.eval_count = data.get("eval_count")
        call.extra = {
            "total_duration_ns": data.get("total_duration"),
            "done_reason": data.get("done_reason"),
        }
        self.calls.append(call)
        return ModelResult(text=text, call=call)

    def call_records(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.calls]


class FakeProvider:
    """Deterministic provider for tests and offline runs.

    Responses are looked up by ``purpose`` (and optionally by a key
    embedded in the prompt). The default returns a JSON object that
    satisfies the lab's verdict shape, so a test can assert on parsing
    behaviour without a network dependency.
    """

    def __init__(
        self,
        model_id: str = "fake-model",
        responses: dict[str, str] | None = None,
        default_response: str | None = None,
    ):
        self.model_id = model_id
        self.responses = dict(responses or {})
        self.default_response = default_response
        self.calls: list[ModelCall] = []
        self.prompts: list[str] = []

    def _lookup(self, prompt: str, purpose: str) -> str:
        if purpose in self.responses:
            return self.responses[purpose]
        for key, value in self.responses.items():
            if key and key in prompt:
                return value
        if self.default_response is not None:
            return self.default_response
        return json.dumps(
            {
                "verdict": "uncertain",
                "explanation": "fake provider default",
                "evidence": [],
            }
        )

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
        started = time.perf_counter()
        self.prompts.append(prompt)
        text = self._lookup(prompt, purpose)
        call = ModelCall(
            call_id=f"fake_{len(self.calls)}",
            model_id=self.model_id,
            prompt_version=prompt_version,
            purpose=purpose,
            latency_ms=(time.perf_counter() - started) * 1000,
            ok=bool(text.strip()),
            response_chars=len(text),
        )
        self.calls.append(call)
        return ModelResult(text=text, call=call)

    def call_records(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.calls]


class UnavailableProvider:
    """Stand-in used when no model endpoint is reachable.

    Every call raises :class:`ProviderError`, and the project code is
    expected to degrade to a deterministic path and record that it did.
    """

    model_id = "unavailable"

    def __init__(self, reason: str = "no model endpoint configured"):
        self.reason = reason
        self.calls: list[ModelCall] = []

    def generate(self, prompt: str, **kwargs: Any) -> ModelResult:
        started = time.perf_counter()
        call = ModelCall(
            call_id=f"unavailable_{len(self.calls)}",
            model_id=self.model_id,
            prompt_version=str(kwargs.get("prompt_version", "v1")),
            purpose=str(kwargs.get("purpose", "generic")),
            latency_ms=(time.perf_counter() - started) * 1000,
            ok=False,
            error=self.reason,
        )
        self.calls.append(call)
        raise ProviderError(self.reason)

    def call_records(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.calls]


def build_provider(
    model_id: str | None = None,
    url: str | None = None,
    *,
    offline: bool = False,
) -> Provider:
    """Create the provider selected by CLI flags or environment."""
    if offline:
        return UnavailableProvider("offline mode requested")
    return OllamaProvider(model_id=model_id, url=url)


def provider_available(provider: Provider) -> tuple[bool, str]:
    """Check whether the provider's endpoint responds. Returns (ok, detail)."""
    if isinstance(provider, OllamaProvider):
        if provider.cache_only:
            return True, f"cache replay configured with {len(provider.cache)} responses; endpoint not checked"
        try:
            with urllib.request.urlopen(provider._endpoint("/api/tags"), timeout=5) as response:
                response.read()
            return True, f"endpoint reachable at {provider.url}"
        except Exception as exc:
            return False, f"endpoint not reachable at {provider.url}: {exc}"
    if isinstance(provider, (FakeProvider,)):
        return True, "fake provider (deterministic)"
    return False, getattr(provider, "reason", "provider unavailable")
