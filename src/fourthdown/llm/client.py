"""The chat model, behind a one-method protocol and one wire format.

Everything downstream depends on `LLMClient`, not on a vendor, so tests substitute a stub
instead of running a model.

`ChatClient` speaks the OpenAI-compatible chat-completions API, which is the closest thing
the ecosystem has to a standard: Ollama (``/v1``), vLLM, llama.cpp's server, and most
hosted providers serve it. Moving from a laptop model to a GPU server or a hosted API is
therefore configuration, not code:

* ``FOURTHDOWN_LLM_BASE_URL`` -- the API root, ending in ``/v1``. Defaults to Ollama at
  ``OLLAMA_HOST`` (``http://localhost:11434``) + ``/v1``.
* ``FOURTHDOWN_LLM_MODEL`` -- the model id at that endpoint (default ``qwen2.5-coder:7b``).
* ``FOURTHDOWN_LLM_API_KEY`` -- sent as a bearer token when set; never logged.
* ``FOURTHDOWN_LLM_TIMEOUT`` -- seconds per completion (default 120).

Defaults suit SQL generation: temperature 0, so the same question produces the same query.
Ollama's context window is set on the server (``OLLAMA_CONTEXT_LENGTH``), not per request.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

import requests

LOGGER = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_TIMEOUT = 120.0
PROBE_TIMEOUT = 5.0


class LLMError(RuntimeError):
    """The model was unreachable or returned something unusable."""


class LLMClient(Protocol):
    """Minimal surface the rest of the project is allowed to depend on."""

    @property
    def model(self) -> str: ...

    def complete(self, *, system: str, prompt: str, temperature: float = 0.0) -> str: ...


def default_base_url() -> str:
    """``FOURTHDOWN_LLM_BASE_URL``, or the OpenAI-compatible root of the Ollama server."""
    explicit = os.environ.get("FOURTHDOWN_LLM_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/") + "/v1"


class ChatClient:
    """An OpenAI-compatible chat-completions client.

    Args:
        model: Model id; ``FOURTHDOWN_LLM_MODEL`` or ``qwen2.5-coder:7b`` when ``None``.
        base_url: API root ending in ``/v1``; see `default_base_url` when ``None``.
        api_key: Bearer token; ``FOURTHDOWN_LLM_API_KEY`` when ``None``, none if unset.
        timeout: Seconds per completion; ``FOURTHDOWN_LLM_TIMEOUT`` or 120 when ``None``.

    `complete` returns the reply text or raises `LLMError`; `available` probes the endpoint.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._model = model or os.environ.get("FOURTHDOWN_LLM_MODEL", DEFAULT_MODEL)
        self._base_url = (base_url or default_base_url()).rstrip("/")
        self._api_key = api_key or os.environ.get("FOURTHDOWN_LLM_API_KEY") or None
        self._timeout = timeout or float(os.environ.get("FOURTHDOWN_LLM_TIMEOUT", DEFAULT_TIMEOUT))

    @property
    def model(self) -> str:
        """The configured model id."""
        return self._model

    @property
    def base_url(self) -> str:
        """The API root requests go to (safe to log: it carries no credentials)."""
        return self._base_url

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def available(self) -> bool:
        """True when the endpoint answers and lists the configured model.

        Returns:
            False if the server is unreachable, rejects the key, or does not serve the
            model (for Ollama: it has not been pulled yet).
        """
        try:
            response = requests.get(
                f"{self._base_url}/models", headers=self._headers(), timeout=PROBE_TIMEOUT
            )
            response.raise_for_status()
            listed = {str(entry.get("id", "")) for entry in response.json().get("data", [])}
        except (requests.RequestException, ValueError, AttributeError):
            return False
        return self._model in listed or f"{self._model}:latest" in listed

    def complete(self, *, system: str, prompt: str, temperature: float = 0.0) -> str:
        """One chat completion.

        Args:
            system: The system message.
            prompt: The user message.
            temperature: Sampling temperature; 0 for deterministic SQL.

        Returns:
            The assistant message text.

        Raises:
            LLMError: The endpoint was unreachable, timed out, returned an error status,
                or replied without content.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "stream": False,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        LOGGER.debug("prompting %s with %d chars", self._model, len(prompt))
        try:
            response = requests.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as error:
            raise LLMError(f"could not reach the LLM at {self._base_url}: {error}") from error
        except ValueError as error:
            raise LLMError(f"the LLM returned invalid JSON: {error}") from error
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None
        if not content:
            raise LLMError(f"the LLM returned no content: {str(body)[:200]}")
        return str(content)


def default_client(model: str | None = None) -> ChatClient:
    """The client configured from the environment, optionally overriding the model."""
    return ChatClient(model)
