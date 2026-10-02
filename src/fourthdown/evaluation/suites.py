"""Every evaluation suite as a function that returns a ``SuiteResult``.

The suites split by what they need. Four run on a GitHub runner with nothing but the
warehouse: the adversarial SQL guard set, keyword routing, the golden reference queries,
and a retrain of every model on a smaller season split. Three need a running model --
text-to-SQL and LLM routing need Ollama's chat model, retrieval needs Postgres plus the
embedding model -- and are skipped with a reason when it is missing.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path

import duckdb

from fourthdown.agent.router import KeywordRouter, LLMRouter, Router
from fourthdown.config import Paths
from fourthdown.data import warehouse
from fourthdown.evaluation import harness, retrieval_harness, routing_harness
from fourthdown.evaluation.scorecard import SuiteResult
from fourthdown.llm import OllamaClient
from fourthdown.models import train as training
from fourthdown.models import winprob
from fourthdown.models.features import DEFAULT_SPLIT, SeasonSplit
from fourthdown.rag.text_to_sql import TextToSQL
from fourthdown.retrieval import DocumentStore, Embedder
from fourthdown.retrieval.embed import EmbeddingError
from fourthdown.retrieval.store import StoreError
from fourthdown.sql import guard

LOGGER = logging.getLogger(__name__)

GUARD_FILE = "golden_guard.json"

GUARD = "guard"
ROUTING_KEYWORD = "routing_keyword"
REFERENCES = "references"
MODELS = "models"
TEXT_TO_SQL = "text_to_sql"
ROUTING_LLM = "routing_llm"
RETRIEVAL = "retrieval"

OFFLINE: tuple[str, ...] = (GUARD, ROUTING_KEYWORD, REFERENCES, MODELS)
"""Suites that need no model server, only (for some) a built warehouse."""

ALL: tuple[str, ...] = (*OFFLINE, TEXT_TO_SQL, ROUTING_LLM, RETRIEVAL)


def _fraction(hits: int, total: int) -> float:
    return hits / total if total else float("nan")


def guard_suite() -> SuiteResult:
    """Attacks must be rejected, legitimate analytics accepted, and every result capped."""
    cases = json.loads(resources.files("fourthdown.evaluation").joinpath(GUARD_FILE).read_text())
    findings: list[str] = []

    attacks = cases["attacks"]
    blocked = 0
    for attack in attacks:
        try:
            guard.validate(attack["sql"])
        except guard.UnsafeSQLError:
            blocked += 1
        else:
            findings.append(f"attack `{attack['id']}` ({attack['category']}) passed validation")

    legitimate = [(case["id"], case["sql"]) for case in cases["legitimate"]] + [
        (f"golden:{question.id}", question.reference_sql)
        for question in harness.load_questions()
        if question.reference_sql
    ]
    allowed = 0
    for case_id, sql in legitimate:
        try:
            guard.validate(sql)
        except guard.UnsafeSQLError as error:
            findings.append(f"legitimate `{case_id}` rejected: {error}")
        else:
            allowed += 1

    caps = cases["row_caps"]
    capped = 0
    for case in caps:
        try:
            limit = guard.validate(case["sql"]).limit
        except guard.UnsafeSQLError as error:
            findings.append(f"row-cap case `{case['id']}` rejected: {error}")
            continue
        if limit == case["limit"]:
            capped += 1
        else:
            findings.append(f"`{case['id']}` limited to {limit}, expected {case['limit']}")

    return SuiteResult(
        GUARD,
        metrics={
            "attacks_blocked": _fraction(blocked, len(attacks)),
            "legitimate_allowed": _fraction(allowed, len(legitimate)),
            "row_caps_enforced": _fraction(capped, len(caps)),
            "attack_cases": float(len(attacks)),
        },
        findings=tuple(findings),
    )


def routing_suite(router: Router, name: str) -> SuiteResult:
    report = routing_harness.evaluate(router, name=name)
    per_tool = [hits / seen for hits, seen in report.by_tool().values() if seen]
    return SuiteResult(
        name,
        metrics={
            "accuracy": report.accuracy(),
            "worst_tool_recall": min(per_tool) if per_tool else float("nan"),
        },
        findings=tuple(
            f"`{result.question.question}` expected {result.question.tool}, got {result.chosen}"
            for result in report.results
            if not result.correct
        ),
    )


def references_suite(connection: duckdb.DuckDBPyConnection) -> SuiteResult:
    """Every hand-written reference query must pass the guard and run on this warehouse."""
    questions = [question for question in harness.load_questions() if question.reference_sql]
    findings: list[str] = []
    runnable = nonempty = 0
    for question in questions:
        assert question.reference_sql is not None
        try:
            rows = connection.execute(guard.validate(question.reference_sql).sql).fetchall()
        except (guard.UnsafeSQLError, duckdb.Error) as error:
            findings.append(f"`{question.id}` failed: {error}")
            continue
        runnable += 1
        if rows:
            nonempty += 1
        else:
            findings.append(f"`{question.id}` returned no rows on this warehouse")
    return SuiteResult(
        REFERENCES,
        metrics={
            "runnable": _fraction(runnable, len(questions)),
            "nonempty": _fraction(nonempty, len(questions)),
        },
        findings=tuple(findings),
    )


def text_to_sql_suite(connection: duckdb.DuckDBPyConnection, client: OllamaClient) -> SuiteResult:
    chain = TextToSQL(connection, client)
    report = harness.evaluate(connection, chain)
    holdout = harness.evaluate(connection, chain, harness.load_questions(holdout=True))
    return SuiteResult(
        TEXT_TO_SQL,
        metrics={
            "correct": _fraction(report.correct, report.total),
            "runnable": _fraction(report.executed, report.total),
            "repairs_per_question": _fraction(report.repairs, report.total),
            "holdout_correct": _fraction(holdout.correct, holdout.total),
            "holdout_runnable": _fraction(holdout.executed, holdout.total),
        },
        findings=tuple(
            f"`{result.question.id}`{' (holdout)' if split else ''}: {result.detail}"
            for split, results in ((False, report.results), (True, holdout.results))
            for result in results
            if not result.correct
        ),
    )


def retrieval_suite(store: DocumentStore, embedder: Embedder, *, k: int) -> SuiteResult:
    report = retrieval_harness.evaluate(store, embedder, k=k)
    dense, lexical = report.mrr("dense"), report.mrr("lexical")
    return SuiteResult(
        RETRIEVAL,
        metrics={
            "hit_at_1": report.hit_at_1(),
            "recall_at_k": report.recall(),
            "mrr": report.mrr(),
            "dense_mrr": dense,
            "lexical_mrr": lexical,
            "hybrid_mrr_gain": report.mrr() - max(dense, lexical),
        },
        findings=tuple(
            f"`{result.question.id}`: {result.detail}"
            for result in report
            if result.hybrid.rank != 1
        ),
    )


def models_suite(
    connection: duckdb.DuckDBPyConnection,
    *,
    split: SeasonSplit,
    epochs: int,
    audit_sample: int,
) -> SuiteResult:
    """Retrain every model from scratch into a scratch directory and score the result.

    Retraining rather than loading checked-in artifacts is the point: it proves the
    pipeline still produces a good model, not that an old file is still on disk.
    """
    with tempfile.TemporaryDirectory() as scratch:
        results = training.run(
            connection,
            model_dir=Path(scratch),
            max_epochs=epochs,
            audit_sample=audit_sample,
            split=split,
        )
    wp, call = results.win_probability, results.play_call
    metrics = {
        "wp_log_loss": wp.log_loss,
        "wp_brier": wp.brier,
        "wp_auc": wp.auc,
        "wp_ece": wp.calibration_error,
        "playcall_accuracy": call.accuracy,
        "playcall_auc": call.auc,
        "playcall_lift_over_majority": call.accuracy - results.play_call_majority.accuracy,
        "advisor_agreement": results.audit.agreement,
        "advisor_go_rate": results.audit.model_go_rate,
        "coach_go_rate": results.audit.coach_go_rate,
    }
    if results.vegas_baseline is not None:
        metrics["wp_log_loss_gap_vs_vegas"] = wp.log_loss - results.vegas_baseline.log_loss
    if results.play_call_xpass is not None:
        metrics["playcall_lift_over_xpass"] = call.accuracy - results.play_call_xpass.accuracy
    return SuiteResult(
        MODELS,
        metrics=metrics,
        findings=(
            f"win probability: {results.wp_dataset.summary()}",
            f"play call: {results.playcall_dataset.summary()}",
            f"fourth-down audit replayed {results.audit.plays:,} held-out plays",
        ),
    )


@dataclass(frozen=True)
class RunOptions:
    paths: Paths
    client: OllamaClient
    embedder: Embedder
    split: SeasonSplit = DEFAULT_SPLIT
    epochs: int = winprob.MAX_EPOCHS
    audit_sample: int = training.FOURTH_DOWN_SAMPLE
    k: int = retrieval_harness.DEFAULT_K


def _with_warehouse(name: str, options: RunOptions) -> SuiteResult:
    if not options.paths.database.exists():
        return SuiteResult(name, skipped="warehouse not built; run `make build`")
    if name == TEXT_TO_SQL and not options.client.available():
        return SuiteResult(name, skipped=f"no {options.client.model} at {options.client.host}")
    with warehouse.connect(options.paths) as connection:
        if name == REFERENCES:
            return references_suite(connection)
        if name == TEXT_TO_SQL:
            return text_to_sql_suite(connection, options.client)
        return models_suite(
            connection,
            split=options.split,
            epochs=options.epochs,
            audit_sample=options.audit_sample,
        )


def _retrieval(options: RunOptions) -> SuiteResult:
    try:
        store = DocumentStore.open()
    except StoreError as error:
        return SuiteResult(RETRIEVAL, skipped=str(error))
    with store:
        if store.count() == 0:
            return SuiteResult(RETRIEVAL, skipped="the document index is empty; run `make index`")
        try:
            options.embedder.encode(["probe"])
        except EmbeddingError as error:
            return SuiteResult(RETRIEVAL, skipped=str(error))
        return retrieval_suite(store, options.embedder, k=options.k)


def _run_one(name: str, options: RunOptions) -> SuiteResult:
    if name == GUARD:
        return guard_suite()
    if name == ROUTING_KEYWORD:
        return routing_suite(KeywordRouter(routing_harness.catalog()), ROUTING_KEYWORD)
    if name == ROUTING_LLM:
        if not options.client.available():
            return SuiteResult(name, skipped=f"no {options.client.model} at {options.client.host}")
        return routing_suite(LLMRouter(routing_harness.catalog(), options.client), ROUTING_LLM)
    if name == RETRIEVAL:
        return _retrieval(options)
    return _with_warehouse(name, options)


def run(selected: Sequence[str], options: RunOptions) -> list[SuiteResult]:
    """Run the selected suites in order, timing each one."""
    unknown = sorted(set(selected) - set(ALL))
    if unknown:
        raise ValueError(f"unknown suite(s) {', '.join(unknown)}; choose from {', '.join(ALL)}")
    results = []
    for name in selected:
        LOGGER.info("running suite %s", name)
        started = time.monotonic()
        result = _run_one(name, options)
        results.append(replace(result, seconds=time.monotonic() - started))
        LOGGER.info("%s finished in %.0fs", name, results[-1].seconds)
    return results
