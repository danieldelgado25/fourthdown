"""Ollama chat client behind a one-method protocol.

Everything downstream depends on `LLMClient`, not on Ollama, so swapping in a hosted API
later is a new class rather than an edit to the SQL chain — and tests substitute a stub
instead of running a model.

Defaults are set for SQL generation: temperature 0 (the same question should produce the
same query) and a short context, since the schema card plus a question is ~1.5k tokens.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Protocol

import requests

LOGGER = logging.getLogger(__name__)

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_TIMEOUT = 180.0


class LLMError(RuntimeError):
    """The model was unreachable or returned something unusable."""


class LLMClient(Protocol):
    """Minimal surface the rest of the project is allowed to depend on."""

    @property
    def model(self) -> str: ...

    def complete(self, *, system: str, prompt: str, temperature: float = 0.0) -> str: ...


class OllamaClient:
    """Talks to a local Ollama server over its HTTP API."""

    def __init__(
        self,
        model: str | None = None,
        *,
        host: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        num_ctx: int = 4096,
    ) -> None:
        self._model = model or os.environ.get("FOURTHDOWN_LLM_MODEL", DEFAULT_MODEL)
        self._host = (host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST)).rstrip("/")
        self._timeout = timeout
        self._num_ctx = num_ctx

    @property
    def model(self) -> str:
        return self._model

    @property
    def host(self) -> str:
        return self._host

    def available(self) -> bool:
        """True when the server answers and has the configured model pulled."""
        try:
            response = requests.get(f"{self._host}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException:
            return False
        tags = {model.get("name", "") for model in response.json().get("models", [])}
        return self._model in tags or f"{self._model}:latest" in tags

    def complete(self, *, system: str, prompt: str, temperature: float = 0.0) -> str:
        payload = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": temperature, "num_ctx": self._num_ctx},
        }
        LOGGER.debug("prompting %s with %d chars", self._model, len(prompt))
        try:
            response = requests.post(f"{self._host}/api/chat", json=payload, timeout=self._timeout)
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as error:
            raise LLMError(
                f"could not reach Ollama at {self._host}: {error}. "
                "Start it with `ollama serve` and pull the model."
            ) from error
        except json.JSONDecodeError as error:
            raise LLMError(f"Ollama returned invalid JSON: {error}") from error
        content = body.get("message", {}).get("content")
        if not content:
            raise LLMError(f"Ollama returned no content: {body}")
        return str(content)


def default_client() -> OllamaClient:
    """The client configured from the environment, used by the CLI."""
    return OllamaClient()
