from __future__ import annotations

from fourthdown.rag.names import NameResolver, Player, hint

ROSTER = [
    Player("00-0038543", "J.Smith-Njigba", "Jaxon Smith-Njigba", "SEA", 2023, 2024),
    Player("00-0031235", "O.Beckham", "Odell Beckham Jr.", "NYG", 2014, 2023),
    Player("00-0027702", "M.Williams", "Mike Williams", "TB", 2010, 2014),
    Player("00-0033536", "M.Williams", "Mike Williams", "LAC", 2017, 2024),
    Player("00-0033873", "P.Mahomes", "Patrick Mahomes", "KC", 2017, 2024),
]
RESOLVER = NameResolver(ROSTER)


def ids(question: str) -> list[list[str]]:
    return [[player.player_id for player in group] for group in RESOLVER.resolve(question)]


def test_finds_a_name_however_it_is_typed():
    for question in (
        "jaxon smith njigba's longest catch?",
        "Jaxon Smith-Njigba yards",
        "jaxon smith-njiba yards",  # a typo
    ):
        assert ids(question) == [["00-0038543"]], question


def test_suffixes_and_several_names():
    assert ids("Odell Beckham vs Patrick Mahomes") == [["00-0031235"], ["00-0033873"]]


def test_a_shared_name_returns_every_candidate():
    assert ids("how did mike williams do in 2018") == [["00-0027702", "00-0033536"]]
    assert "pick by team and season" in hint(RESOLVER.resolve("mike williams"))


def test_questions_without_a_player_get_no_hint():
    assert ids("Which team loves to pass against the Bills on third and long?") == []
    assert hint([]) == ""
