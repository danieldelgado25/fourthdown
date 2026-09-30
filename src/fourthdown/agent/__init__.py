"""Tool routing: one question in, one tool's evidence out."""

from fourthdown.agent.assistant import Answer, Assistant
from fourthdown.agent.build import Readiness, Services, services
from fourthdown.agent.router import KeywordRouter, LLMRouter, Route, Router
from fourthdown.agent.tools import (
    AdvisorTool,
    Evidence,
    NarrativeTool,
    Passage,
    StatsTool,
    Table,
    TendencyTool,
    Tool,
)

__all__ = [
    "AdvisorTool",
    "Answer",
    "Assistant",
    "Evidence",
    "KeywordRouter",
    "LLMRouter",
    "NarrativeTool",
    "Passage",
    "Readiness",
    "Route",
    "Router",
    "Services",
    "StatsTool",
    "Table",
    "TendencyTool",
    "Tool",
    "services",
]
