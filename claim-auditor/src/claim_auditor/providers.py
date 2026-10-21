"""Model provider adapters.

One adapter talks to an Ollama-compatible HTTP endpoint; one is a
deterministic fake used by the tests and by the no-model path.

Rules enforced here:

* No API key is required or read. The endpoint is configurable by
  environment variable or CLI flag, so a hosted provider can replace a
  local one without touching the auditor.
* Every call records the model id, the prompt version, and the
  latency.
* ``generate`` is synchronous. Callers should issue one request at a
  time unless the endpoint's concurrency has been verified.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "minimax-m3.1-flash:cloud"

ENV_ENDPOINT = "CLAIM_AUDITOR_OLLAMA_URL"
ENV_MODEL = "CLAIM_AUDITOR_MODEL"


class ProviderError(RuntimeError):
    """Raised when the endpoint is unreachable or returns an error."""


class ParseError(RuntimeError):
    """Raised when a response cannot be parsed into the expected shape."""

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
    prompt_chars: int = 0
    response_chars: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "purpose": self.purpose,
            "latency_ms": round(self.latency_ms, 2),
            "ok": self.ok,
            "prompt_chars": self.prompt_chars,
            "response_chars": self.response_chars,
            "error": self.error,
        }


@dataclass
class ModelResult:
    """The result of a model call: the text plus the call record."""

    text: str
    call: ModelCall

    def json(self) -> Any:
        return parse_json_block(self.text, raw=self.text)


class Provider(Protocol):
    """The minimal provider interface the auditor needs."""

    model_id: str

    def generate(
        self,
        prompt: str,
        *,
        purpose: str = "generic",
        prompt_version: str = "v1",
        temperature: float = 0.0,
        json_mode: bool = False,
        timeout: float = 180.0,
    ) -> ModelResult:
        ...


def parse_json_block(text: str, raw: str | None = None) -> Any:
    """Parse JSON from a model response.

    Tolerates a fenced code block and surrounding prose, because models
    frequently wrap JSON even when told not to.
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


class OllamaProvider:
    """Adapter for an Ollama-compatible HTTP endpoint."""

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
        self.calls: list[ModelCall] = []

    def _post(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ProviderError(f"HTTP {exc.code} from {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"cannot reach {self.url}{path}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise ProviderError(f"non-JSON response from {path}: {exc}") from exc

    def generate(
        self,
        prompt: str,
        *,
        purpose: str = "generic",
        prompt_version: str = "v1",
        temperature: float = 0.0,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> ModelResult:
        started = time.perf_counter()
        messages = [{"role": "user", "content": prompt}]
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
            prompt_chars=len(prompt),
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
        self.calls.append(call)
        return ModelResult(text=text, call=call)

    def call_records(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.calls]


class FakeProvider:
    """A deterministic provider for tests and offline runs."""

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
        return json.dumps({"verdict": "uncertain", "explanation": "fake provider default"})

    def generate(
        self,
        prompt: str,
        *,
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
