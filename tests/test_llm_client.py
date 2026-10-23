"""The OpenAI-compatible chat client, against a stubbed transport."""

from __future__ import annotations

from typing import Any

import pytest
import requests

from fourthdown.llm import client as llm
from fourthdown.llm.client import ChatClient, LLMError


class Reply:
    def __init__(self, body: Any, status: int = 200) -> None:
        self._body = body
        self.status_code = status

    def json(self) -> Any:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


@pytest.fixture()
def calls(monkeypatch) -> list[dict[str, Any]]:
    recorded: list[dict[str, Any]] = []

    def post(url: str, **kwargs: Any) -> Reply:
        recorded.append({"url": url, **kwargs})
        return Reply({"choices": [{"message": {"content": "SELECT 1"}}]})

    monkeypatch.setattr(llm.requests, "post", post)
    return recorded


def test_complete_speaks_the_chat_completions_api(calls):
    reply = ChatClient("m", base_url="http://x/v1/", api_key="k").complete(
        system="sys", prompt="q", temperature=0.2
    )
    assert reply == "SELECT 1"
    call = calls[0]
    assert call["url"] == "http://x/v1/chat/completions"
    assert call["headers"] == {"Authorization": "Bearer k"}
    assert call["json"]["model"] == "m" and call["json"]["temperature"] == 0.2
    assert [m["role"] for m in call["json"]["messages"]] == ["system", "user"]


def test_no_key_sends_no_authorization_header(calls, monkeypatch):
    monkeypatch.delenv("FOURTHDOWN_LLM_API_KEY", raising=False)
    ChatClient("m", base_url="http://x/v1").complete(system="s", prompt="q")
    assert calls[0]["headers"] == {}


def test_base_url_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("FOURTHDOWN_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
    assert ChatClient().base_url == "http://ollama:11434/v1"
    monkeypatch.setenv("FOURTHDOWN_LLM_BASE_URL", "https://api.example.com/v1")
    assert ChatClient().base_url == "https://api.example.com/v1"


def test_errors_and_empty_replies_raise(monkeypatch):
    client = ChatClient("m", base_url="http://x/v1")
    monkeypatch.setattr(llm.requests, "post", lambda url, **kw: Reply({}, status=500))
    with pytest.raises(LLMError):
        client.complete(system="s", prompt="q")
    monkeypatch.setattr(llm.requests, "post", lambda url, **kw: Reply({"choices": []}))
    with pytest.raises(LLMError, match="no content"):
        client.complete(system="s", prompt="q")


def test_available_requires_the_model_to_be_listed(monkeypatch):
    listing = Reply({"data": [{"id": "qwen2.5-coder:7b"}, {"id": "nomic-embed-text:latest"}]})
    monkeypatch.setattr(llm.requests, "get", lambda url, **kw: listing)
    assert ChatClient("qwen2.5-coder:7b", base_url="http://x/v1").available()
    assert ChatClient("nomic-embed-text", base_url="http://x/v1").available()
    assert not ChatClient("llama3", base_url="http://x/v1").available()

    def down(url: str, **kw: Any) -> Reply:
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(llm.requests, "get", down)
    assert not ChatClient("qwen2.5-coder:7b", base_url="http://x/v1").available()
