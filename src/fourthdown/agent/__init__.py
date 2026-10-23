"""Tool routing: one question in, one tool's evidence out.

Everything exported here is free of heavy dependencies, so the gateway service can route
without DuckDB, Postgres, or torch installed. Concrete tools live in `fourthdown.services`.
"""

from fourthdown.agent.assistant import Answer, Assistant, answer_json
from fourthdown.agent.router import KeywordRouter, LLMRouter, Route, Router
from fourthdown.agent.tools import Evidence, Passage, Table, Tool, ToolError

__all__ = [
    "Answer",
    "Assistant",
    "Evidence",
    "KeywordRouter",
    "LLMRouter",
    "Passage",
    "Route",
    "Router",
    "Table",
    "Tool",
    "ToolError",
    "answer_json",
]
