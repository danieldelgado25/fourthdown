"""LLM access. Ollama locally today; the protocol is the seam for anything else."""

from fourthdown.llm.client import LLMClient, LLMError, OllamaClient, default_client

__all__ = ["LLMClient", "LLMError", "OllamaClient", "default_client"]
