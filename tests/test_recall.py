from __future__ import annotations

from conftest import ScriptedClient

from fourthdown.retrieval import recall
from fourthdown.retrieval.store import Hit

HIT = Hit(
    doc_id="game:2014_21_NE_SEA",
    grain="game",
    game_id="2014_21_NE_SEA",
    season=2014,
    week=21,
    teams=("NE", "SEA"),
    title="2014 Super Bowl XLIX (49): New England Patriots at Seattle Seahawks",
    body="New England Patriots beat Seattle Seahawks by 4.",
    score=0.03,
    dense_rank=1,
    lexical_rank=2,
)


class StubRetriever:
    def __init__(self, *hits: Hit) -> None:
        self._hits = list(hits)
        self.calls: list[dict[str, object]] = []

    def search(self, question: str, **kwargs: object) -> list[Hit]:
        self.calls.append({"question": question, **kwargs})
        return self._hits


def test_passages_are_numbered_for_citation() -> None:
    passages = recall.format_passages([HIT, HIT])
    assert passages.startswith("[1] 2014 Super Bowl")
    assert "[2] 2014 Super Bowl" in passages


def test_answer_grounds_the_prompt_in_the_hits() -> None:
    retriever = StubRetriever(HIT)
    client = ScriptedClient("New England won by four [1].")

    answer = recall.answer(retriever, client, "Who won Super Bowl XLIX?", k=3, season=2014)

    assert answer.answer == "New England won by four [1]."
    assert answer.hits == [HIT]
    assert retriever.calls == [
        {
            "question": "Who won Super Bowl XLIX?",
            "k": 3,
            "grain": None,
            "season": 2014,
            "team": None,
        }
    ]
    prompt = client.prompts[0]
    assert "[1] 2014 Super Bowl" in prompt
    assert "Who won Super Bowl XLIX?" in prompt
    assert client.temperatures == [0.1]


def test_answer_renders_its_sources() -> None:
    answer = recall.answer(
        StubRetriever(HIT), ScriptedClient("Patriots [1]."), "Who won Super Bowl XLIX?"
    )
    rendered = answer.render()
    assert "Sources:" in rendered
    assert "[1] 2014 Super Bowl XLIX (49)" in rendered
    assert "both" in rendered, "a hit from both lists should say so"


def test_no_passages_means_no_model_call() -> None:
    client = ScriptedClient()
    answer = recall.answer(StubRetriever(), client, "Who won the 1974 Super Bowl?")
    assert answer.hits == []
    assert "Nothing in the narrative index" in answer.answer
    assert client.prompts == []
