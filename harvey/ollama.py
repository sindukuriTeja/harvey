"""Ollama — a local-model brain provider for Harvey.

Runs Harvey's thinking on a model served by a local Ollama instance
(https://ollama.com). No subscription, no API key, no per-token bill: the
model runs on the operator's own machine.

This is a community addition on top of the upstream Harvey brain layer,
which ships with ``claude`` (Claude Code) and ``codex`` (Codex CLI) as its
providers. The interface mirrors harvey/codex.py (``OllamaError`` with a
``kind``/``retryable`` pair, an ``OllamaResult`` dataclass, an
``OllamaClient`` with ``run``/``preflight``/``fingerprint``) so the Brain
dispatches to it the same way it dispatches to Codex.

Configuration (harvey.yaml)::

    brain:
      provider: ollama
      model: qwen2.5:14b        # any model in `ollama list`
      base_url: http://127.0.0.1:11434   # optional; this is the default

The Ollama server is expected to be started by the operator (``ollama
serve``); Harvey never starts or stops it.

Honest limits, stated here so the dashboard and the README can say them:
local models are far weaker at long, structured agentic work than Claude
or GPT. Expect more JSON parse failures, weaker personalization, and
slower calls (minutes, not seconds, on a laptop). Harvey's JSON extraction
is forgiving, but quality is on the operator.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger("harvey.ollama")

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
# Suggested model: strong enough for structured JSON, small enough for a
# laptop GPU or a fast CPU. The operator can pick anything in `ollama list`.
SUGGESTED_MODEL = "qwen2.5:14b"

# Local generation is slow: a single call may take minutes on a laptop.
# The Brain's DEFAULT_TIMEOUT_SECONDS (300s) applies per call; this is only
# the preflight's own ceiling.
PREFLIGHT_TIMEOUT_SECONDS = 30.0
REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)

# Ollama error strings that mean "retrying now is pointless" (the model is
# not pulled, the server is down, etc.). Transient ones (429, 5xx, a busy
# GPU) are retryable.
_NON_RETRYABLE_HINTS = (
    "not found",           # model not pulled
    "model not found",
    "connection refused",
    "no such host",
    "not logged in",
)


class OllamaError(Exception):
    """An Ollama call failed. ``kind`` drives retry and the operator message.

    Messages never contain prompts or replies.
    """

    def __init__(self, message: str, kind: str = "protocol", retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


def _setup_error(message: str) -> OllamaError:
    return OllamaError(message, kind="setup")


def _normalize_base_url(value: str) -> str:
    """Validate/normalize a base URL from harvey.yaml (brain.base_url).

    Precedence: explicit brain.base_url, then OLLAMA_BASE_URL in the
    environment, then the default.
    """
    value = (value or "").strip().rstrip("/")
    if not value:
        value = os.environ.get("OLLAMA_BASE_URL", "").strip().rstrip("/")
    if not value:
        return DEFAULT_BASE_URL
    if not value.startswith(("http://", "https://")):
        raise _setup_error(f"Ollama base URL must start with http:// or https:// (got '{value}').")
    return value


@dataclass
class OllamaResult:
    text: str
    model: str = ""
    thread_id: str = ""
    duration_ms: int = 0
    usage: dict = field(default_factory=dict)
    rate_limits: dict | None = None


class OllamaClient:
    """Talks to a local Ollama server with single-shot, tool-less prompts.

    Harvey's prompts are self-contained (system context inlined), so one
    ``/api/chat`` call per ``think()`` is all the provider needs. No
    streaming, no tools, no function calling: the response is text.
    """

    def __init__(self, model: str = "", base_url: str = ""):
        self.model = model or SUGGESTED_MODEL
        self.base_url = _normalize_base_url(base_url)
        self._lock: asyncio.Lock | None = None

    def _serialize(self) -> asyncio.Lock:
        # One in-flight generation at a time per Brain: local GPUs can't
        # serve two Harvey calls at once, and queuing keeps tokens honest.
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def fingerprint(self) -> tuple:
        """What a paused provider is worth re-checking against. Ollama has
        no login file, so this is just the configured endpoint + model."""
        return (self.base_url, self.model)

    async def preflight(self, timeout: float = PREFLIGHT_TIMEOUT_SECONDS) -> None:
        """Everything a real call checks, short of generating: the server
        answers, and the configured model is pulled. No prompt is sent, so
        nothing is inferred or metered as a model call. Raises OllamaError."""
        try:
            async with asyncio.timeout(timeout):
                async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                    # 1) Server reachable.
                    try:
                        resp = await client.get(f"{self.base_url}/api/version")
                    except httpx.ConnectError as e:
                        raise _setup_error(
                            f"Ollama server not reachable at {self.base_url}. "
                            "Start it with 'ollama serve' (it usually runs "
                            "automatically after 'ollama --version' on a "
                            "desktop install), then restart Harvey."
                        ) from e
                    except httpx.HTTPError as e:
                        raise OllamaError(
                            f"Ollama server at {self.base_url} failed to answer "
                            f"({type(e).__name__}).",
                            kind="runtime", retryable=True,
                        ) from e
                    if resp.status_code != 200:
                        raise OllamaError(
                            f"Ollama server answered {resp.status_code} at {self.base_url}.",
                            kind="runtime", retryable=True,
                        )

                    # 2) Model pulled.
                    try:
                        resp = await client.get(f"{self.base_url}/api/tags")
                    except httpx.HTTPError as e:
                        raise OllamaError(
                            f"Ollama server at {self.base_url} failed to list models "
                            f"({type(e).__name__}).",
                            kind="runtime", retryable=True,
                        ) from e
                    if resp.status_code != 200:
                        raise OllamaError(
                            f"Ollama /api/tags answered {resp.status_code}.",
                            kind="runtime", retryable=True,
                        )
                    try:
                        payload = resp.json()
                    except ValueError as e:
                        raise OllamaError("Ollama /api/tags returned no JSON.",
                                          kind="protocol", retryable=True) from e
                    models = [
                        str(m.get("name") or "")
                        for m in (payload.get("models") or [])
                        if isinstance(m, dict)
                    ]
                    if self.model not in models:
                        # Ollama also accepts tagless names ("qwen2.5" matches
                        # "qwen2.5:14b"); check the name part too.
                        base_name = self.model.split(":")[0]
                        if not any(name.split(":")[0] == base_name for name in models):
                            shown = ", ".join(models[:8]) or "(none)"
                            raise _setup_error(
                                f"Model '{self.model}' is not pulled on this Ollama "
                                f"server. Available: {shown}. "
                                f"Pull it with 'ollama pull {self.model}'."
                            )
        except asyncio.TimeoutError:
            raise OllamaError(
                f"Ollama preflight timed out after {timeout:.0f}s at {self.base_url}.",
                kind="runtime", retryable=True,
            ) from None

    async def run(self, prompt: str, timeout: float, skip_if=None) -> OllamaResult:
        """One generation. ``skip_if`` (returns a reason or "") is checked
        while holding the lock, mirroring CodexClient.run."""
        deadline = time.monotonic() + timeout
        try:
            async with asyncio.timeout(timeout):
                async with self._serialize():
                    reason = skip_if() if skip_if else ""
                    if reason:
                        raise OllamaError(reason, kind="paused")
                    return await self._generate(prompt, deadline)
        except OllamaError:
            raise
        except asyncio.TimeoutError:
            raise OllamaError(
                f"Ollama generation timed out after {timeout:.0f}s "
                f"(local models are slow; try a smaller model).",
                kind="runtime", retryable=True,
            ) from None
        except httpx.HTTPError as e:
            raise OllamaError(
                f"Ollama request failed ({type(e).__name__}).",
                kind="runtime", retryable=True,
            ) from None

    async def _generate(self, prompt: str, deadline: float) -> OllamaResult:
        started = time.monotonic()
        body = {
            "model": self.model,
            "stream": False,
            # Harvey's prompts already carry their own system context; keep
            # the model in a plain assistant role.
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                # Local models drift into prose around JSON; a modest
                # temperature keeps output stable without freezing it.
                "temperature": 0.4,
                # Local context is precious; cap generation so a runaway
                # answer can't eat the whole window (Harvey's prompts ask
                # for short outputs anyway).
                "num_predict": 4096,
            },
        }
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=body)

        if resp.status_code == 404:
            raise _setup_error(
                f"Ollama answered 404 for model '{self.model}' — it is not "
                f"pulled. Run 'ollama pull {self.model}'."
            )
        if resp.status_code == 429:
            raise OllamaError("Ollama is busy (429); another generation is running.",
                              kind="runtime", retryable=True)
        if resp.status_code >= 500:
            raise OllamaError(f"Ollama server error ({resp.status_code}).",
                              kind="runtime", retryable=True)
        if resp.status_code != 200:
            raise OllamaError(
                f"Ollama answered {resp.status_code}: {resp.text[:200]}",
                kind="protocol", retryable=False,
            )

        try:
            payload = resp.json()
        except ValueError as e:
            raise OllamaError("Ollama returned no JSON.", kind="protocol",
                              retryable=True) from e

        message = payload.get("message") or {}
        text = str(message.get("content") or "").strip()
        if payload.get("error"):
            raise OllamaError(
                f"Ollama error: {str(payload['error'])[:200]}",
                kind="protocol", retryable=False,
            )
        if not text:
            raise OllamaError("Ollama returned an empty response.",
                              kind="runtime", retryable=True)

        usage = {
            "input_tokens": int(payload.get("prompt_eval_count") or 0),
            "output_tokens": int(payload.get("eval_count") or 0),
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            # Local: no per-token bill.
            "cost_usd": 0.0,
        }
        return OllamaResult(
            text=text,
            model=str(payload.get("model") or self.model),
            thread_id=f"ollama-{int(started * 1000)}",
            duration_ms=int((time.monotonic() - started) * 1000),
            usage=usage,
        )


class OllamaQuotaClient:
    """No subscription quota to read for a local model.

    ``get_utilization`` returns None, which makes the Brain fall back to
    Harvey's own daily call counter (the ``max_daily_claude_percent``
    budget still applies, against that counter).
    """

    def __init__(self, client: OllamaClient):
        self._client = client

    async def get_utilization(self) -> Optional[dict]:
        return None

    def invalidate(self) -> None:  # interface parity with CodexQuotaClient
        pass