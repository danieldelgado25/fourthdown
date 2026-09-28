"""Pull the structured bits out of an English question, without asking a model.

Slot filling is the one part of the pipeline where an LLM buys nothing: "4th and 2 from
the 38, down 3 with four minutes left" is a regex, and a regex cannot invent a yard line
that was never said. Anything genuinely ambiguous returns `None` and the caller asks for
it rather than guessing.
"""

from __future__ import annotations

import re

from fourthdown.models.winprob import GameState
from fourthdown.narrative.teams import TEAM_NAMES

DEFAULT_SEASON = 2024
DEFAULT_MINUTES = 15.0

WORD_NUMBERS: dict[str, float] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
    "inches": 0.5,
    "goal": 0.0,
}

_NUMBER = r"(\d+(?:\.\d+)?|" + "|".join(WORD_NUMBERS) + ")"
_SEASON = re.compile(r"\b(19[9]\d|20[0-4]\d)\b")
_TOGO = re.compile(rf"\b(?:4th|fourth)\s*(?:and|&)\s*{_NUMBER}", re.IGNORECASE)
_YARDLINE = re.compile(
    r"\b(?:from|at|on)\s+(?:the\s+)?(?:opponent'?s?\s+)?(\d{1,2})[- ]?(?:yard[- ]?line)?\b",
    re.IGNORECASE,
)
_MINUTES = re.compile(rf"{_NUMBER}\s*(?:minutes?|mins?)\b", re.IGNORECASE)
_SECONDS = re.compile(rf"{_NUMBER}\s*(?:seconds?|secs?)\b", re.IGNORECASE)
_MARGIN = re.compile(rf"\b(down|trailing by|up|leading by|ahead by|behind by)\s+{_NUMBER}", re.I)

_NICKNAMES: dict[str, str] = {
    name.rsplit(" ", 1)[-1].lower(): abbreviation for abbreviation, name in TEAM_NAMES.items()
}
_CITIES: dict[str, str] = {
    name.rsplit(" ", 1)[0].lower(): abbreviation for abbreviation, name in TEAM_NAMES.items()
}


def _number(token: str) -> float:
    return WORD_NUMBERS[token.lower()] if token.lower() in WORD_NUMBERS else float(token)


def parse_season(question: str, *, default: int = DEFAULT_SEASON) -> int:
    """The last four-digit year that looks like a season, or the default."""
    found = _SEASON.findall(question)
    return int(found[-1]) if found else default


def parse_team(question: str) -> str | None:
    """A team abbreviation, matched on nickname, city, or the abbreviation itself."""
    words = re.findall(r"[A-Za-z0-9']+", question)
    for word in words:
        if word.upper() in TEAM_NAMES:
            return word.upper()
    lowered = [word.lower() for word in words]
    for word in lowered:
        if word in _NICKNAMES:
            return _NICKNAMES[word]
    text = " ".join(lowered)
    for city, abbreviation in _CITIES.items():
        if city in text:
            return abbreviation
    return None


def parse_state(question: str) -> GameState | None:
    """A fourth-down situation, or `None` when distance or field position is missing.

    Those two are required because every option's value depends on them; the clock and
    the score have defensible defaults (a tied game in the fourth quarter) and the
    caller sees what was assumed in the rendered situation line.
    """
    togo = _TOGO.search(question)
    yardline = _YARDLINE.search(question)
    if togo is None or yardline is None:
        return None
    distance = _number(togo.group(1))
    spot = float(yardline.group(1))
    return GameState(
        yardline_100=spot,
        down=4,
        ydstogo=distance,
        game_seconds_remaining=_parse_clock(question),
        score_differential=_parse_margin(question),
        goal_to_go=spot <= distance,
    )


def _parse_clock(question: str) -> float:
    minutes = _MINUTES.search(question)
    seconds = _SECONDS.search(question)
    if minutes is None and seconds is None:
        return DEFAULT_MINUTES * 60.0
    total = _number(minutes.group(1)) * 60.0 if minutes else 0.0
    total += _number(seconds.group(1)) if seconds else 0.0
    return total


def _parse_margin(question: str) -> float:
    match = _MARGIN.search(question)
    if match is None:
        return 0.0
    magnitude = _number(match.group(2))
    trailing = match.group(1).lower() in {"down", "trailing by", "behind by"}
    return -magnitude if trailing else magnitude
