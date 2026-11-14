"""Find the players a question names before any SQL is written.

People type 'jaxon smith njigba'; the plays store 'J.Smith-Njigba', an abbreviation that
197 names share with another player. Matching the question against full names here is
deterministic, and hands the model the player_id, which is unique, instead of leaving it
to guess the spelling.
"""

from __future__ import annotations

import difflib
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import duckdb

SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})
# High enough that 'the bills' does not become a player; low enough for one typo.
FUZZY_CUTOFF = 0.9
MIN_FUZZY_CHARS = 9

ROSTER_SQL = """
SELECT player_id, any_value(player_name), any_value(full_name),
       string_agg(DISTINCT team, '/' ORDER BY team), min(season), max(season)
FROM player_game
WHERE full_name IS NOT NULL
GROUP BY player_id
"""


@dataclass(frozen=True)
class Player:
    player_id: str
    player_name: str
    full_name: str
    teams: str
    first_season: int
    last_season: int


def _words(text: str) -> list[str]:
    text = re.sub(r"['’]s\b", "", text.lower())
    return re.findall(r"[a-z0-9]+", re.sub(r"['’.]", "", text))


def _key(full_name: str) -> str:
    return " ".join(word for word in _words(full_name) if word not in SUFFIXES)


def load_roster(connection: duckdb.DuckDBPyConnection) -> list[Player]:
    with connection.cursor() as cursor:
        return [Player(*row) for row in cursor.execute(ROSTER_SQL).fetchall()]


class NameResolver:
    def __init__(self, roster: Iterable[Player]) -> None:
        self._by_key: dict[str, list[Player]] = defaultdict(list)
        for player in roster:
            self._by_key[_key(player.full_name)].append(player)
        self._keys = list(self._by_key)

    def resolve(self, question: str) -> list[list[Player]]:
        """One group per name in the question; a group holds every player it could mean."""
        words = _words(question)
        used: set[int] = set()
        found: list[tuple[int, list[Player]]] = []
        for size in (4, 3, 2):  # longest first, so 'odell beckham jr' is not 'odell beckham'
            for start in range(len(words) - size + 1):
                span = set(range(start, start + size))
                if span & used:
                    continue
                window = " ".join(words[start : start + size])
                players = self._by_key.get(window)
                if players is None and len(window) >= MIN_FUZZY_CHARS:
                    close = difflib.get_close_matches(window, self._keys, n=1, cutoff=FUZZY_CUTOFF)
                    players = self._by_key[close[0]] if close else None
                if players:
                    used |= span
                    found.append((start, players))
        return [players for _, players in sorted(found, key=lambda item: item[0])]


def hint(groups: list[list[Player]]) -> str:
    """The lines appended to the question; empty when it names nobody."""
    if not groups:
        return ""
    lines = ["Players named in the question (player_id is unique, the abbreviation is not):"]
    for players in groups:
        if len(players) > 1:
            name = players[0].full_name
            lines.append(f"- {name} is more than one player; pick by team and season:")
        lines += [
            f"- {p.full_name}: player_id '{p.player_id}', stored as '{p.player_name}' "
            f"({p.teams} {p.first_season}-{p.last_season})"
            for p in players
        ]
    return "\n".join(lines) + "\n"
