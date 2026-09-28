"""Turn warehouse rows into short English documents.

Retrieval needs prose, and the warehouse has none: a play table stores `posteam = 'KC'`
and `fixed_drive_result = 'Touchdown'`, which no cosine similarity will match against
"what happened on the Chiefs' last drive". So the corpus is *generated*, deterministically,
from the same views the SQL layer queries.

Two grains, for two kinds of question:

- one document per game, which is what "how did that game go" needs;
- one document per drive, which is what "what happened on the final drive" needs.

Numbers are rendered into the text but the documents are never the source of an answer to
a numeric question -- that is text-to-SQL's job. They exist to locate *which* game or
drive the user means, and to give the model something to narrate from.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal

import duckdb

from fourthdown.narrative.teams import name, nickname

LOGGER = logging.getLogger(__name__)

GRAINS: tuple[str, ...] = ("game", "drive")

PLAYOFF_ROUNDS: tuple[str, ...] = (
    "Wild Card round",
    "Divisional round",
    "Conference Championship",
    "Super Bowl",
)

ROMAN: tuple[tuple[int, str], ...] = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)

QUARTERS: dict[int, str] = {
    1: "first quarter",
    2: "second quarter",
    3: "third quarter",
    4: "fourth quarter",
    5: "overtime",
}


@dataclass(frozen=True)
class Document:
    """A retrievable passage and the metadata retrieval filters on."""

    doc_id: str
    grain: str
    game_id: str
    season: int
    week: int
    teams: tuple[str, ...]
    title: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.title}. {self.body}"


def _roman(number: int) -> str:
    out = []
    for value, numeral in ROMAN:
        count, number = divmod(number, value)
        out.append(numeral * count)
    return "".join(out)


def week_label(season: int, season_type: str, week: int) -> str:
    """Name the round, because nobody asks about "2014 week 21"."""
    if season_type != "POST":
        return f"Week {week}"
    first_post_week = 18 if season <= 2020 else 19
    index = min(max(week - first_post_week, 0), len(PLAYOFF_ROUNDS) - 1)
    round_name = PLAYOFF_ROUNDS[index]
    if round_name != "Super Bowl":
        return round_name
    number = season - 1965
    # Super Bowl 50 was branded with the digits, every other one with a numeral.
    return f"Super Bowl {number}" if number == 50 else f"Super Bowl {_roman(number)} ({number})"


def _number(value: object) -> float | None:
    """DuckDB hands back ints, floats, Decimals and Nones in the same column."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    return None


def _yardline(yardline_100: object, defteam: str) -> str:
    spot = _number(yardline_100)
    if spot is None:
        return "an unknown spot"
    yards = int(spot)
    if yards > 50:
        return f"their own {100 - yards}"
    if yards == 50:
        return "midfield"
    return f"the {nickname(defteam)} {yards}"


def _rate(value: object) -> str:
    number = _number(value)
    return "unknown" if number is None else f"{number:.0%}"


def _epa(value: object) -> str:
    number = _number(value)
    return "unknown" if number is None else f"{number:+.2f}"


def _conditions(row: dict[str, object]) -> str:
    roof = row["roof"]
    parts: list[str] = []
    if isinstance(roof, str) and roof in {"dome", "closed"}:
        parts.append("indoors")
    temp, wind = row["temp"], row["wind"]
    if isinstance(temp, (int, float)):
        parts.append(f"{int(temp)}F")
    if isinstance(wind, (int, float)) and wind:
        parts.append(f"{int(wind)} mph wind")
    return ", ".join(parts)


def _season_filter(seasons: Sequence[int] | None, alias: str = "") -> tuple[str, list[int]]:
    if not seasons:
        return "", []
    prefix = f"{alias}." if alias else ""
    placeholders = ", ".join("?" for _ in seasons)
    return f" WHERE {prefix}season IN ({placeholders})", list(seasons)


def _rows(
    connection: duckdb.DuckDBPyConnection, sql: str, params: Sequence[int]
) -> Iterator[dict[str, object]]:
    cursor = connection.execute(sql, list(params))
    columns = [description[0] for description in cursor.description or ()]
    while batch := cursor.fetchmany(2000):
        for row in batch:
            yield dict(zip(columns, row, strict=True))


GAME_SQL = """
SELECT g.*,
       h.pass_rate AS home_pass_rate, h.epa_per_play AS home_epa,
       h.success_rate AS home_success, h.yards_gained AS home_yards,
       h.third_down_conversions AS home_third_made, h.third_down_attempts AS home_third_att,
       h.interceptions + h.fumbles_lost AS home_turnovers,
       a.pass_rate AS away_pass_rate, a.epa_per_play AS away_epa,
       a.success_rate AS away_success, a.yards_gained AS away_yards,
       a.third_down_conversions AS away_third_made, a.third_down_attempts AS away_third_att,
       a.interceptions + a.fumbles_lost AS away_turnovers
FROM games g
JOIN team_game h ON h.game_id = g.game_id AND h.team = g.home_team
JOIN team_game a ON a.game_id = g.game_id AND a.team = g.away_team
{where}
ORDER BY g.season, g.week, g.game_id
"""

LEADERS_SQL = """
SELECT game_id, team, role, player_name, plays, yards, touchdowns
FROM player_game
{where}
QUALIFY row_number() OVER (PARTITION BY game_id, team, role ORDER BY yards DESC) = 1
"""


def _leaders(
    connection: duckdb.DuckDBPyConnection, seasons: Sequence[int] | None
) -> dict[tuple[str, str], list[str]]:
    where, params = _season_filter(seasons)
    collected: dict[tuple[str, str], list[str]] = {}
    for row in _rows(connection, LEADERS_SQL.format(where=where), params):
        player = row["player_name"]
        if not isinstance(player, str):
            continue
        verb = {"passer": "threw for", "rusher": "ran for", "receiver": "caught"}[str(row["role"])]
        noun = "yards" if row["role"] != "receiver" else "receiving yards"
        phrase = f"{player} {verb} {row['yards']} {noun}"
        touchdowns = row["touchdowns"]
        if isinstance(touchdowns, int) and touchdowns:
            phrase += f" and {touchdowns} touchdown{'s' if touchdowns > 1 else ''}"
        collected.setdefault((str(row["game_id"]), str(row["team"])), []).append(phrase)
    return collected


def _team_line(row: dict[str, object], side: str, team: str) -> str:
    attempts = row[f"{side}_third_att"]
    third = (
        f"converted {row[f'{side}_third_made']} of {attempts} third downs"
        if isinstance(attempts, int) and attempts
        else "faced no third downs"
    )
    turnovers = row[f"{side}_turnovers"]
    giveaways = (
        "did not turn it over"
        if turnovers == 0
        else f"turned it over {turnovers} time{'' if turnovers == 1 else 's'}"
    )
    return (
        f"{name(team)} passed on {_rate(row[f'{side}_pass_rate'])} of designed plays, "
        f"gained {row[f'{side}_yards']} yards at {_epa(row[f'{side}_epa'])} EPA per play, "
        f"{third}, and {giveaways}"
    )


def game_documents(
    connection: duckdb.DuckDBPyConnection, seasons: Iterable[int] | None = None
) -> Iterator[Document]:
    """One document per game: result, conditions, both teams' efficiency, and leaders."""
    wanted = sorted(seasons) if seasons is not None else None
    where, params = _season_filter(wanted, alias="g")
    leaders = _leaders(connection, wanted)
    for row in _rows(connection, GAME_SQL.format(where=where), params):
        game_id, home, away = str(row["game_id"]), str(row["home_team"]), str(row["away_team"])
        season, week = int(str(row["season"])), int(str(row["week"]))
        label = week_label(season, str(row["season_type"]), week)
        home_score, away_score = row["home_score"], row["away_score"]
        margin = row["home_margin"]
        if isinstance(margin, int) and margin != 0:
            winner, loser = (home, away) if margin > 0 else (away, home)
            outcome = f"{name(winner)} beat {name(loser)} by {abs(margin)}"
        else:
            outcome = f"{name(home)} and {name(away)} tied"
        conditions = _conditions(row)
        setting = f"{row['stadium']}" + (f", {conditions}" if conditions else "")
        title = f"{season} {label}: {name(away)} at {name(home)}"
        sentences = [
            f"Played {row['game_date']} at {setting}",
            f"Final score {name(away)} {away_score}, {name(home)} {home_score}; {outcome}",
            _team_line(row, "away", away),
            _team_line(row, "home", home),
        ]
        sentences += [
            f"{name(team)} leaders: {'; '.join(leaders[(game_id, team)])}"
            for team in (away, home)
            if (game_id, team) in leaders
        ]
        body = ". ".join(sentences)
        yield Document(
            doc_id=f"game:{game_id}",
            grain="game",
            game_id=game_id,
            season=season,
            week=week,
            teams=(away, home),
            title=title,
            body=body + ".",
        )


DRIVE_SQL = """
WITH best AS (
    SELECT drive_id,
           arg_max(play_desc, epa) AS best_play,
           max(epa) AS best_epa
    FROM plays
    WHERE drive_id IS NOT NULL AND epa IS NOT NULL{season_clause}
    GROUP BY drive_id
)
SELECT d.*, g.season_type, g.home_team, g.away_team, g.game_date,
       best.best_play, best.best_epa
FROM drives d
JOIN games g ON g.game_id = d.game_id
LEFT JOIN best ON best.drive_id = d.drive_id
{where}
ORDER BY d.game_id, d.drive_number
"""


def drive_documents(
    connection: duckdb.DuckDBPyConnection,
    seasons: Iterable[int] | None = None,
    *,
    postseason_only: bool = False,
) -> Iterator[Document]:
    """One document per drive: where it started, what it did, how it ended.

    ``postseason_only`` exists because drives outnumber games 23 to 1 and embedding is the
    slow step; playoff drives are both the ones people ask about and a 4% slice.
    """
    wanted = sorted(seasons) if seasons is not None else None
    where, params = _season_filter(wanted, alias="d")
    if postseason_only:
        where = f"{where} AND g.season_type = 'POST'" if where else " WHERE g.season_type = 'POST'"
    season_clause = f" AND season IN ({', '.join('?' for _ in wanted)})" if wanted else ""
    sql = DRIVE_SQL.format(where=where, season_clause=season_clause)
    for row in _rows(connection, sql, list(wanted or []) + params):
        game_id = str(row["game_id"])
        posteam, defteam = str(row["posteam"]), str(row["defteam"])
        season, week = int(str(row["season"])), int(str(row["week"]))
        label = week_label(season, str(row["season_type"]), week)
        quarter = QUARTERS.get(int(str(row["start_quarter"])), "overtime")
        result = str(row["drive_result"] or "unknown").lower()
        start = _yardline(row["start_yardline_100"], defteam)
        first_downs = row["first_downs"] or 0
        title = (
            f"{season} {label}, {name(posteam)} drive {row['drive_number']} against {name(defteam)}"
        )
        sentences = [
            f"{quarter.capitalize()}: {name(posteam)} started at {start}",
            f"{row['plays']} plays, {row['yards_gained']} yards, "
            f"{first_downs} first down{'' if first_downs == 1 else 's'}, "
            f"{_epa(row['epa_per_play'])} EPA per play",
            f"Result: {result}",
        ]
        best_play = row["best_play"]
        if isinstance(best_play, str):
            sentences.append(f"Best play ({_epa(row['best_epa'])} EPA): {best_play.strip()}")
        yield Document(
            doc_id=f"drive:{row['drive_id']}",
            grain="drive",
            game_id=game_id,
            season=season,
            week=week,
            teams=(posteam, defteam),
            title=title,
            body=". ".join(sentences) + ".",
        )


def documents(
    connection: duckdb.DuckDBPyConnection,
    *,
    grains: Sequence[str] = GRAINS,
    seasons: Iterable[int] | None = None,
    postseason_drives_only: bool = False,
) -> Iterator[Document]:
    """Every requested grain, in one stream."""
    wanted = sorted(seasons) if seasons is not None else None
    for grain in grains:
        if grain not in GRAINS:
            raise ValueError(f"unknown grain {grain!r}; expected one of {GRAINS}")
    if "game" in grains:
        yield from game_documents(connection, wanted)
    if "drive" in grains:
        yield from drive_documents(connection, wanted, postseason_only=postseason_drives_only)
