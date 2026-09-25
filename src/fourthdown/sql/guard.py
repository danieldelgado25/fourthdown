"""Static validation of model-generated SQL before it reaches DuckDB.

The threat here is not a malicious user — it is a language model that will cheerfully
write ``DROP TABLE``, invent a ``player_season`` view, or ask for 768,085 rows. Parsing
with sqlglot rather than pattern-matching strings means the checks see the same structure
DuckDB will: a comment cannot hide a second statement, and ``deleteme`` is not a DELETE.

Read-only connections catch writes too, but a parse-time rejection gives the repair loop
an error message specific enough to fix, and never spends a query on the warehouse.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from fourthdown.data.warehouse import VIEW_NAMES

DIALECT = "duckdb"
DEFAULT_ROW_LIMIT = 200

ALLOWED_VIEWS: frozenset[str] = frozenset(VIEW_NAMES)

# Anything that writes, reads the filesystem, or reaches outside the warehouse.
FORBIDDEN_NODES: tuple[type[exp.Expr], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.TruncateTable,
    exp.Copy,
    exp.Attach,
    exp.Command,  # sqlglot's catch-all for statements it does not model (PRAGMA, SET, ...)
)

# DuckDB table functions that read arbitrary paths or URLs.
FORBIDDEN_FUNCTIONS: frozenset[str] = frozenset(
    {"read_parquet", "read_csv", "read_csv_auto", "read_json", "read_json_auto", "glob"}
)


class UnsafeSQLError(ValueError):
    """Raised when generated SQL fails validation. The message goes back to the model."""


@dataclass(frozen=True)
class ValidatedSQL:
    """A query that parsed, passed every check, and carries a row limit."""

    sql: str
    tables: frozenset[str]
    limit: int


def _statement(sql: str) -> exp.Expr:
    try:
        statements = [
            statement for statement in sqlglot.parse(sql, dialect=DIALECT) if statement is not None
        ]
    except sqlglot.ParseError as error:  # pragma: no cover - message varies by sqlglot version
        raise UnsafeSQLError(f"query does not parse: {error}") from error
    if not statements:
        raise UnsafeSQLError("no SQL statement found")
    if len(statements) > 1:
        raise UnsafeSQLError("only one statement is allowed; found multiple separated by ';'")
    parsed: exp.Expr = statements[0]
    return parsed


def _as_query(statement: exp.Expr) -> exp.Query:
    for node_type in FORBIDDEN_NODES:
        if isinstance(statement, node_type) or statement.find(node_type):
            raise UnsafeSQLError(
                f"only read-only SELECT queries are allowed; found {node_type.key.upper()}"
            )
    if not isinstance(statement, exp.Query):
        raise UnsafeSQLError("query must be a SELECT")
    return statement


def _referenced_tables(statement: exp.Expr) -> frozenset[str]:
    """Table names excluding CTE aliases, which are defined by the query itself."""
    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    names = {
        table.name.lower()
        for table in statement.find_all(exp.Table)
        if table.name and table.name.lower() not in cte_names
    }
    return frozenset(names)


def _check_tables(tables: frozenset[str]) -> None:
    unknown = sorted(tables - ALLOWED_VIEWS)
    if unknown:
        raise UnsafeSQLError(
            f"unknown table(s) {', '.join(unknown)}; "
            f"query only these views: {', '.join(sorted(ALLOWED_VIEWS))}"
        )
    if not tables:
        raise UnsafeSQLError("query does not read any of the warehouse views")


def _check_functions(statement: exp.Expr) -> None:
    """Reject file-reading functions, which sqlglot models as nodes or unknown calls."""
    for function in statement.find_all(exp.Func):
        names = (
            [function.name] if isinstance(function, exp.Anonymous) else list(function.sql_names())
        )
        for name in names:
            if name.lower() in FORBIDDEN_FUNCTIONS:
                raise UnsafeSQLError(f"{name.lower()}() is not allowed; query the views instead")


def _apply_limit(statement: exp.Query, max_rows: int) -> tuple[exp.Query, int]:
    """Cap the result size, tightening an existing LIMIT rather than replacing it."""
    limit = statement.args.get("limit")
    if limit is None:
        return statement.limit(max_rows), max_rows
    requested = limit.expression
    if isinstance(requested, exp.Literal) and requested.is_int:
        value = int(requested.name)
        if value <= max_rows:
            return statement, value
    return statement.limit(max_rows), max_rows


def validate(sql: str, *, max_rows: int = DEFAULT_ROW_LIMIT) -> ValidatedSQL:
    """Parse and check generated SQL, returning it normalised and row-limited.

    Raises ``UnsafeSQLError`` with a message written to be handed straight back to the
    model as repair feedback.
    """
    statement = _as_query(_statement(sql))
    _check_functions(statement)
    tables = _referenced_tables(statement)
    _check_tables(tables)
    limited, limit = _apply_limit(statement, max_rows)
    return ValidatedSQL(sql=limited.sql(dialect=DIALECT, pretty=True), tables=tables, limit=limit)
