"""Postgres + pgvector document store with hybrid (dense + lexical) retrieval.

Postgres holds both halves of the search on purpose. The dense half is an HNSW index over
`vector(n)`; the lexical half is a generated `tsvector` with a GIN index. Keeping them in
one table means one filter (`season`, `grain`, `teams`) applies to both, one query fuses
them, and there is no second service to keep consistent with the first.

Fusion is reciprocal rank fusion:

    score(doc) = 1 / (k + dense_rank) + 1 / (k + lexical_rank)

RRF is used rather than a weighted sum of similarities because the two halves produce
incomparable numbers -- cosine distance and `ts_rank_cd` have no shared scale -- and
tuning a weight between them on a 20-question evaluation set would be fitting noise.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import psycopg

from fourthdown.narrative import Document

LOGGER = logging.getLogger(__name__)

DEFAULT_DSN = "postgresql://fourthdown:fourthdown@localhost:5432/fourthdown"
RRF_K = 60
"""The constant in reciprocal rank fusion; 60 is the value from the original paper."""

DEFAULT_CANDIDATES = 50
UPSERT_BATCH = 200
MODES: tuple[str, ...] = ("hybrid", "dense", "lexical")


class StoreError(RuntimeError):
    """Raised when the store is unreachable or configured for different embeddings."""


def dsn() -> str:
    return os.environ.get("FOURTHDOWN_PG_DSN", DEFAULT_DSN)


@dataclass(frozen=True)
class Hit:
    """A retrieved document and the ranks that put it there."""

    doc_id: str
    grain: str
    game_id: str
    season: int
    week: int
    teams: tuple[str, ...]
    title: str
    body: str
    score: float
    dense_rank: int | None
    lexical_rank: int | None

    @property
    def text(self) -> str:
        return f"{self.title}. {self.body}"

    @property
    def found_by(self) -> str:
        if self.dense_rank is not None and self.lexical_rank is not None:
            return "both"
        return "dense" if self.dense_rank is not None else "lexical"


SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id      text PRIMARY KEY,
    grain       text NOT NULL,
    game_id     text NOT NULL,
    season      integer NOT NULL,
    week        integer NOT NULL,
    teams       text[] NOT NULL,
    title       text NOT NULL,
    body        text NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS
                (to_tsvector('english', title || ' ' || body)) STORED,
    embedding   vector({dimensions}) NOT NULL
);

CREATE INDEX IF NOT EXISTS documents_tsv ON documents USING gin (tsv);
CREATE INDEX IF NOT EXISTS documents_grain_season ON documents (grain, season);
CREATE INDEX IF NOT EXISTS documents_game ON documents (game_id);
CREATE INDEX IF NOT EXISTS documents_embedding
    ON documents USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS embedding_space (
    id          integer PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    model       text NOT NULL,
    dimensions  integer NOT NULL
);
"""

SEARCH = """
WITH filtered AS (
    SELECT * FROM documents
    WHERE (%(grain)s::text IS NULL OR grain = %(grain)s)
      AND (%(season)s::int IS NULL OR season = %(season)s)
      AND (%(team)s::text IS NULL OR %(team)s = ANY (teams))
), dense AS (
    SELECT doc_id, row_number() OVER (ORDER BY embedding <=> %(vector)s::vector) AS rank
    FROM filtered
    WHERE %(use_dense)s::bool
    ORDER BY embedding <=> %(vector)s::vector
    LIMIT %(candidates)s
), lexical AS (
    SELECT doc_id, row_number() OVER (ORDER BY ts_rank_cd(tsv, parsed.query) DESC) AS rank
    -- ORed rather than ANDed: a question is a sentence, and requiring every
    -- content word ("happened", "between") matches nothing. ts_rank_cd still
    -- rewards the documents that cover more of the query.
    FROM filtered,
         (
             SELECT nullif(
                 replace(plainto_tsquery('english', %(question)s)::text, ' & ', ' | '), ''
             )::tsquery AS query
         ) AS parsed
    WHERE %(use_lexical)s::bool AND tsv @@ parsed.query
    ORDER BY ts_rank_cd(tsv, parsed.query) DESC
    LIMIT %(candidates)s
)
SELECT d.doc_id, d.grain, d.game_id, d.season, d.week, d.teams, d.title, d.body,
       dense.rank AS dense_rank, lexical.rank AS lexical_rank,
       coalesce(1.0 / (%(rrf_k)s + dense.rank), 0)
     + coalesce(1.0 / (%(rrf_k)s + lexical.rank), 0) AS score
FROM documents d
JOIN (SELECT doc_id FROM dense UNION SELECT doc_id FROM lexical) hit USING (doc_id)
LEFT JOIN dense USING (doc_id)
LEFT JOIN lexical USING (doc_id)
ORDER BY score DESC, d.doc_id
LIMIT %(k)s
"""

UPSERT = """
INSERT INTO documents (doc_id, grain, game_id, season, week, teams, title, body, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
ON CONFLICT (doc_id) DO UPDATE SET
    grain = EXCLUDED.grain, game_id = EXCLUDED.game_id, season = EXCLUDED.season,
    week = EXCLUDED.week, teams = EXCLUDED.teams, title = EXCLUDED.title,
    body = EXCLUDED.body, embedding = EXCLUDED.embedding
"""


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in vector) + "]"


class DocumentStore:
    """Owns the schema and every query against it."""

    def __init__(self, connection: psycopg.Connection) -> None:
        self._connection = connection

    @classmethod
    def open(cls, url: str | None = None) -> DocumentStore:
        try:
            connection = psycopg.connect(url or dsn(), autocommit=True)
        except psycopg.Error as error:
            raise StoreError(
                f"could not connect to Postgres at {url or dsn()}: {error}. "
                "Start it with `docker compose up -d`."
            ) from error
        return cls(connection)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> DocumentStore:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def initialise(self, *, model: str, dimensions: int, reset: bool = False) -> None:
        """Create the schema, or rebuild it when the embedding space changed."""
        current = self.embedding_space()
        if current is not None and current != (model, dimensions):
            if not reset:
                raise StoreError(
                    f"the index holds {current[0]} vectors of {current[1]} dimensions, but "
                    f"{model} produces {dimensions}. Re-index with --reset."
                )
            LOGGER.info("embedding space changed from %s to %s; dropping documents", current, model)
            reset = True
        if reset:
            self._connection.execute("DROP TABLE IF EXISTS documents")
            self._connection.execute("DROP TABLE IF EXISTS embedding_space")
        self._connection.execute(SCHEMA.format(dimensions=dimensions))
        self._connection.execute(
            "INSERT INTO embedding_space (id, model, dimensions) VALUES (1, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET model = EXCLUDED.model, "
            "dimensions = EXCLUDED.dimensions",
            (model, dimensions),
        )

    def embedding_space(self) -> tuple[str, int] | None:
        if not self._table_exists("embedding_space"):
            return None
        row = self._connection.execute(
            "SELECT model, dimensions FROM embedding_space WHERE id = 1"
        ).fetchone()
        if row is None:
            return None
        return str(row[0]), int(row[1])

    def _table_exists(self, table: str) -> bool:
        row = self._connection.execute("SELECT to_regclass(%s) IS NOT NULL", (table,)).fetchone()
        return bool(row and row[0])

    def upsert(self, documents: Iterable[tuple[Document, Sequence[float]]]) -> int:
        """Insert or replace documents; returns how many rows were written."""
        written = 0
        batch: list[tuple[object, ...]] = []
        for document, vector in documents:
            batch.append(
                (
                    document.doc_id,
                    document.grain,
                    document.game_id,
                    document.season,
                    document.week,
                    list(document.teams),
                    document.title,
                    document.body,
                    _vector_literal(vector),
                )
            )
            if len(batch) >= UPSERT_BATCH:
                written += self._flush(batch)
                batch = []
        written += self._flush(batch)
        return written

    def _flush(self, batch: list[tuple[object, ...]]) -> int:
        if not batch:
            return 0
        with self._connection.cursor() as cursor:
            cursor.executemany(UPSERT, batch)
        return len(batch)

    def count(self, *, grain: str | None = None) -> int:
        row = self._connection.execute(
            "SELECT count(*) FROM documents WHERE (%s::text IS NULL OR grain = %s)",
            (grain, grain),
        ).fetchone()
        return int(row[0]) if row else 0

    def coverage(self) -> list[tuple[str, int, int, int]]:
        """(grain, documents, first season, last season) for the report and the CLI."""
        rows = self._connection.execute(
            "SELECT grain, count(*), min(season), max(season) FROM documents "
            "GROUP BY grain ORDER BY grain"
        ).fetchall()
        return [(str(row[0]), int(row[1]), int(row[2]), int(row[3])) for row in rows]

    def search(
        self,
        question: str,
        embedding: Sequence[float],
        *,
        k: int = 5,
        grain: str | None = None,
        season: int | None = None,
        team: str | None = None,
        mode: str = "hybrid",
        candidates: int = DEFAULT_CANDIDATES,
    ) -> list[Hit]:
        """Dense and lexical candidate lists fused by reciprocal rank.

        ``mode`` switches a half off (``dense`` or ``lexical``), which is what the
        retrieval evaluation needs to show the fusion is earning its place.
        """
        if mode not in MODES:
            raise ValueError(f"unknown search mode {mode!r}; expected one of {MODES}")
        rows = self._connection.execute(
            SEARCH,
            {
                "use_dense": mode in {"hybrid", "dense"},
                "use_lexical": mode in {"hybrid", "lexical"},
                "vector": _vector_literal(embedding),
                "question": question,
                "grain": grain,
                "season": season,
                "team": team,
                "candidates": candidates,
                "rrf_k": RRF_K,
                "k": k,
            },
        ).fetchall()
        return list(_hits(rows))


def _teams(value: object) -> tuple[str, ...]:
    """``text[]`` comes back as a list, but the row is typed loosely."""
    return tuple(str(team) for team in value) if isinstance(value, list) else ()


def _hits(rows: Iterable[tuple[object, ...]]) -> Iterator[Hit]:
    for row in rows:
        yield Hit(
            doc_id=str(row[0]),
            grain=str(row[1]),
            game_id=str(row[2]),
            season=int(str(row[3])),
            week=int(str(row[4])),
            teams=_teams(row[5]),
            title=str(row[6]),
            body=str(row[7]),
            dense_rank=None if row[8] is None else int(str(row[8])),
            lexical_rank=None if row[9] is None else int(str(row[9])),
            score=float(str(row[10])),
        )
