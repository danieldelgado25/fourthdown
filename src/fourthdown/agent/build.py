"""Assemble an assistant from whatever this machine actually has.

Four tools need four different backing services -- DuckDB, Postgres, Ollama, trained
model files -- and a developer who has only run `fourthdown build` has one of them. So
each tool is attached only if its dependency answers, and what was skipped (and why) is
reported rather than swallowed: the dashboard shows it, and a question routed to a
missing tool says "not loaded" instead of dying.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field

import duckdb

from fourthdown.agent.assistant import Assistant
from fourthdown.agent.router import KeywordRouter, LLMRouter, Router
from fourthdown.agent.tools import AdvisorTool, NarrativeTool, StatsTool, TendencyTool, Tool
from fourthdown.config import Paths, data_paths
from fourthdown.data import warehouse
from fourthdown.llm import OllamaClient
from fourthdown.models import fourth_down
from fourthdown.models.train import FOURTH_DOWN_ARTIFACT, WP_ARTIFACT
from fourthdown.models.winprob import WinProbabilityModel
from fourthdown.rag.text_to_sql import TextToSQL
from fourthdown.retrieval import DocumentStore, Retriever, default_embedder
from fourthdown.retrieval.store import StoreError

LOGGER = logging.getLogger(__name__)


@dataclass
class Readiness:
    """Which backing services answered, for `/api/health` and the dashboard banner."""

    warehouse: bool = False
    llm: bool = False
    retrieval: bool = False
    models: bool = False
    tools: tuple[str, ...] = ()
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        return bool(self.skipped)


@dataclass
class Services:
    """Everything the API holds for the life of the process.

    The advisor and the connection are exposed alongside the assistant because two
    endpoints take structured input (a fourth-down state, a season) and should not have
    to phrase it back into English for the router to re-parse.
    """

    assistant: Assistant
    readiness: Readiness
    advisor: fourth_down.FourthDownAdvisor | None = None
    connection: duckdb.DuckDBPyConnection | None = None


@contextmanager
def services(
    *,
    paths: Paths | None = None,
    model: str | None = None,
) -> Iterator[Services]:
    """Open every service that is available and yield the assistant built from them."""
    resolved = paths or data_paths()
    readiness = Readiness()
    tools: list[Tool] = []
    client = OllamaClient(model)
    readiness.llm = client.available()
    if not readiness.llm:
        readiness.skipped["llm"] = f"Ollama has no {client.model} at {client.host}"

    with ExitStack() as stack:
        connection = None
        try:
            connection = stack.enter_context(warehouse.connect(resolved))
        except (FileNotFoundError, duckdb.Error) as error:
            readiness.skipped["warehouse"] = str(error)
        if connection is not None:
            readiness.warehouse = True
            tools.append(TendencyTool(connection))
            if readiness.llm:
                tools.append(StatsTool(TextToSQL(connection, client)))
            else:
                readiness.skipped["stats"] = "text-to-SQL needs the LLM"

        if readiness.llm:
            try:
                store = stack.enter_context(DocumentStore.open())
                tools.append(NarrativeTool(Retriever(store, default_embedder()), client))
                readiness.retrieval = True
            except StoreError as error:
                readiness.skipped["retrieval"] = str(error)
        else:
            readiness.skipped["retrieval"] = "grounded answers need the LLM"

        advisor = _load_advisor(resolved)
        if advisor is None:
            readiness.skipped["models"] = f"no trained artifacts in {resolved.models}"
        else:
            readiness.models = True
            tools.append(AdvisorTool(advisor))

        if not tools:
            raise RuntimeError(f"nothing is available to answer with: {readiness.skipped}")
        readiness.tools = tuple(tool.name for tool in tools)
        yield Services(
            assistant=Assistant(tools, _router(tools, client, readiness)),
            readiness=readiness,
            advisor=advisor,
            connection=connection,
        )


def _router(tools: list[Tool], client: OllamaClient, readiness: Readiness) -> Router:
    return LLMRouter(tools, client) if readiness.llm else KeywordRouter(tools)


def _load_advisor(paths: Paths) -> fourth_down.FourthDownAdvisor | None:
    weights = paths.models / WP_ARTIFACT
    components = paths.models / FOURTH_DOWN_ARTIFACT
    if not weights.exists() or not components.exists():
        return None
    return fourth_down.load_components(components, WinProbabilityModel.load(weights))
