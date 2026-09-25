from __future__ import annotations

import pytest
import requests

from fourthdown.retrieval import embed


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    embedder = embed.HashingEmbedder(dimensions=64)
    first, second = embedder.encode(["fourth and one"] * 2)
    assert first == second
    assert len(first) == embedder.dimensions == 64
    assert sum(value * value for value in first) == pytest.approx(1.0)
    assert embedder.encode([""])[0] == [0.0] * 64


def test_hashing_embedder_separates_different_text() -> None:
    embedder = embed.HashingEmbedder(dimensions=64)
    touchdown, punt = embedder.encode(["touchdown", "punt"])
    assert touchdown != punt


def test_ollama_embedder_batches_and_reports_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_post(url: str, json: dict[str, object], timeout: float) -> FakeResponse:
        calls.append({"url": url, "json": json})
        inputs = json["input"]
        assert isinstance(inputs, list)
        return FakeResponse({"embeddings": [[0.5, 0.5] for _ in inputs]})

    monkeypatch.setattr(requests, "post", fake_post)
    embedder = embed.OllamaEmbedder("test-model", host="http://ollama:11434/")
    vectors = embedder.encode([f"text {index}" for index in range(embed.BATCH_SIZE + 1)])

    assert len(vectors) == embed.BATCH_SIZE + 1
    assert len(calls) == 2, "inputs beyond the batch size should be a second request"
    assert calls[0]["url"] == "http://ollama:11434/api/embed"
    assert embedder.name == "ollama:test-model"
    assert embedder.dimensions == 2


def test_ollama_embedder_reports_an_unreachable_server(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, json: dict[str, object], timeout: float) -> FakeResponse:
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", fake_post)
    with pytest.raises(embed.EmbeddingError, match="could not reach Ollama"):
        embed.OllamaEmbedder(host="http://localhost:1").encode(["hello"])


def test_ollama_embedder_rejects_a_truncated_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda url, json, timeout: FakeResponse({"embeddings": [[0.1]]}),
    )
    with pytest.raises(embed.EmbeddingError, match="for 2 inputs"):
        embed.OllamaEmbedder().encode(["one", "two"])


def test_embedders_satisfy_the_protocol() -> None:
    assert isinstance(embed.HashingEmbedder(), embed.Embedder)
    assert isinstance(embed.default_embedder(), embed.Embedder)
