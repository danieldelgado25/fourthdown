"""Which filter emptied a query: rerun it with each WHERE condition removed in turn.

An empty result says nothing about why. Counting the rows the FROM clause yields with all
filters, then with each top-level ``AND`` condition left out, names the culprit directly:
if dropping ``season_type = 'POST'`` brings back 14 rows and dropping anything else
brings back none, the playoff filter is the problem. The counts are cheap DuckDB queries
and need no model.

Only the outermost SELECT's WHERE is probed, split on ``AND``. A condition that is an
``OR`` group stays one condition. Set operations (UNION) and queries without a WHERE have
nothing to probe and return no probes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import duckdb
import sqlglot
from sqlglot import exp

from fourthdown.sql.guard import DIALECT

LOGGER = logging.getLogger(__name__)

MAX_CONDITIONS = 8
"""More conditions than this is a query too tangled to probe usefully; the rest are skipped."""


@dataclass(frozen=True)
class Probe:
    """Rows the FROM clause yields with one condition removed (``None`` if that failed)."""

    condition: str
    rows_without: int | None


@dataclass(frozen=True)
class FilterReport:
    """Rows matched with every filter, and with each one removed."""

    rows_with_all: int | None
    probes: tuple[Probe, ...]

    def render(self) -> str:
        """Lines for a prompt or a log: the condition and what removing it leaves."""
        lines = [f"rows matching every filter: {_count(self.rows_with_all)}"]
        lines += [
            f"without {probe.condition}: {_count(probe.rows_without)}" for probe in self.probes
        ]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, object]:
        return {
            "rows_with_all": self.rows_with_all,
            "probes": {probe.condition: probe.rows_without for probe in self.probes},
        }


def _count(rows: int | None) -> str:
    return "query failed" if rows is None else f"{rows} rows"


def _select(sql: str) -> exp.Select | None:
    try:
        parsed = sqlglot.parse_one(sql, dialect=DIALECT)
    except sqlglot.errors.ParseError:
        return None
    return parsed if isinstance(parsed, exp.Select) else None


def conditions(sql: str) -> list[exp.Expr]:
    """The top-level ``AND`` conditions of the outermost SELECT's WHERE, in order."""
    select = _select(sql)
    where = select.args.get("where") if select is not None else None
    if where is None:
        return []
    found = list(where.this.flatten()) if isinstance(where.this, exp.And) else [where.this]
    # flatten() unwraps parentheses; an OR group shown without them reads as a different filter.
    return [exp.Paren(this=c) if isinstance(c, exp.Connector) else c for c in found]


def counting_query(sql: str, keep: list[exp.Expr]) -> str | None:
    """``sql`` reduced to ``count(*)`` over its FROM and joins, filtered by ``keep``.

    Aggregates, grouping, ordering, and limits are dropped: what matters is how many source
    rows survive the filters, not what the original query made of them.
    """
    select = _select(sql)
    if select is None:
        return None
    counting = select.copy()
    for arg in ("where", "group", "having", "order", "limit", "offset", "qualify", "distinct"):
        counting.set(arg, None)
    counting.set("expressions", [exp.Count(this=exp.Star())])
    if keep:
        counting = counting.where(exp.and_(*(condition.copy() for condition in keep)))
    return counting.sql(dialect=DIALECT)


def _run_count(connection: duckdb.DuckDBPyConnection, sql: str | None) -> int | None:
    if sql is None:
        return None
    try:
        with connection.cursor() as cursor:
            row = cursor.execute(sql).fetchone()
    except duckdb.Error as error:
        LOGGER.info("filter probe failed: %s", error)
        return None
    return int(row[0]) if row and row[0] is not None else None


def probe(connection: duckdb.DuckDBPyConnection, sql: str) -> FilterReport | None:
    """Count source rows with every filter and with each one removed; ``None`` if no WHERE."""
    found = conditions(sql)[:MAX_CONDITIONS]
    if not found:
        return None
    rendered = [condition.sql(dialect=DIALECT) for condition in found]
    probes = tuple(
        Probe(
            condition=rendered[index],
            rows_without=_run_count(
                connection, counting_query(sql, found[:index] + found[index + 1 :])
            ),
        )
        for index in range(len(found))
    )
    rows_with_all = _run_count(connection, counting_query(sql, found))
    return FilterReport(rows_with_all=rows_with_all, probes=probes)
