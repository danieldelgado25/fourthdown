"""Current club names for the normalised abbreviations the ETL writes."""

from __future__ import annotations

TEAM_NAMES: dict[str, str] = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}
"""Abbreviation -> club name.

The narratives spell the name out because retrieval is asked questions in English:
"what happened in the Seahawks' final drive" has to match a document that says Seahawks,
not one that says ``SEA``.
"""


def name(abbreviation: str | None) -> str:
    if not abbreviation:
        return "unknown"
    return TEAM_NAMES.get(abbreviation, abbreviation)


def nickname(abbreviation: str | None) -> str:
    """The club nickname alone: ``KC`` -> ``Chiefs``."""
    return name(abbreviation).rsplit(" ", 1)[-1]
