"""LLM access over the OpenAI-compatible chat API: Ollama, vLLM, or a hosted provider."""

from fourthdown.llm.client import ChatClient, LLMClient, LLMError, default_client

__all__ = ["ChatClient", "LLMClient", "LLMError", "default_client"]
