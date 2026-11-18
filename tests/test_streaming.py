"""The streaming replay: events, pacing, scoring, delivery guarantees, and the feed. All on
the in-memory broker; the Kafka adapter's settings are checked without a broker."""

from __future__ import annotations

import json

import duckdb
import pytest

from fourthdown.streaming import MemoryBroker, feed, replay, scorer
from fourthdown.streaming.broker import consumer_config, producer_config
from fourthdown.streaming.events import (
    DEAD_LETTER_TOPIC,
    PLAYS_TOPIC,
    WP_TOPIC,
    EventError,
    GameFinal,
    PlayEvent,
    decode_game_event,
    decode_win_probability,
    encode,
)


def _play(sequence: int, *, clock: float, posteam_is_home: bool = True, **changes) -> PlayEvent:
    values = dict(
        game_id="2023_22_SF_KC",
        play_id=100 + sequence,
        sequence=sequence,
        home_team="KC",
        away_team="SF",
        posteam="KC" if posteam_is_home else "SF",
        qtr=1,
        down=1,
        ydstogo=10.0,
        yardline_100=75.0,
        goal_to_go=False,
        game_seconds_remaining=clock,
        half_seconds_remaining=max(clock - 1800, 0),
        score_differential=0.0,
        posteam_timeouts_remaining=3,
        defteam_timeouts_remaining=3,
        posteam_is_home=posteam_is_home,
        posteam_spread=0.0,
        home_score=0,
        away_score=0,
        description=f"play {sequence}",
        nflfastr_home_wp=0.5,
    )
    values.update(changes)
    return PlayEvent(**values)


GAME = [
    _play(0, clock=3600),
    _play(1, clock=3570, posteam_is_home=False),
    _play(2, clock=3000),
    GameFinal("2023_22_SF_KC", 3, "KC", "SF", home_score=25, away_score=22),
]


def offense_wins(chance: float):
    """A stand-in model: the offense wins with ``chance``, whatever the situation."""
    return scorer.Scorer(probability=lambda state: chance, model_version="test")


def _pipeline(events, model, broker=None):
    broker = broker or MemoryBroker()
    replay.replay(broker.producer(), events, speed=0)
    stats = scorer.run(
        broker.consumer([PLAYS_TOPIC], scorer.GROUP), broker.producer(), model, idle_seconds=0
    )
    live = feed.watch(broker.consumer([WP_TOPIC], feed.GROUP), feed.Feed(), emit=lambda line: None)
    return broker, stats, live


# --- events -----------------------------------------------------------------------------------


def test_events_round_trip_through_json():
    for event in GAME:
        assert decode_game_event(encode(event)) == event


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (b"\xff\xfe", "not JSON"),
        (b"[1, 2]", "not a JSON object"),
        (json.dumps({"v": 99, "kind": "play"}).encode(), "schema version 99"),
        (json.dumps({"v": 1, "kind": "punt"}).encode(), "unknown kind 'punt'"),
        (json.dumps({"v": 1, "kind": "final", "game_id": "x"}).encode(), "missing"),
    ],
)
def test_malformed_messages_are_refused_with_a_reason(value, reason):
    with pytest.raises(EventError, match=reason):
        decode_game_event(value)


# --- the replay ---------------------------------------------------------------------------


@pytest.fixture()
def game_tables():
    """The two semantic tables the replay reads, holding one short game: a kickoff (no
    down), three scrimmage plays out of order, and a timeout (no team with the ball)."""
    connection = duckdb.connect()
    connection.execute(
        """
        CREATE TABLE plays AS SELECT * FROM (VALUES
            (1, NULL, 'KC', 3600, 0, 'kickoff'),
            (40, 2, 'SF', 3500, -3, 'second'),
            (12, 1, 'KC', 3590, 0, 'first'),
            (55, NULL, NULL, 3480, NULL, 'timeout'),
            (61, 3, 'SF', 3450, -3, 'third')
        ) AS t(play_id, down, posteam, game_seconds_remaining, score_differential, play_desc)
        """
    )
    connection.execute(
        """
        ALTER TABLE plays ADD COLUMN game_id VARCHAR DEFAULT '2023_22_SF_KC';
        ALTER TABLE plays ADD COLUMN home_team VARCHAR DEFAULT 'KC';
        ALTER TABLE plays ADD COLUMN away_team VARCHAR DEFAULT 'SF';
        ALTER TABLE plays ADD COLUMN qtr INTEGER DEFAULT 1;
        ALTER TABLE plays ADD COLUMN ydstogo INTEGER DEFAULT 10;
        ALTER TABLE plays ADD COLUMN yardline_100 INTEGER DEFAULT 75;
        ALTER TABLE plays ADD COLUMN goal_to_go BOOLEAN DEFAULT false;
        ALTER TABLE plays ADD COLUMN half_seconds_remaining INTEGER DEFAULT 1700;
        ALTER TABLE plays ADD COLUMN posteam_timeouts_remaining INTEGER DEFAULT 3;
        ALTER TABLE plays ADD COLUMN defteam_timeouts_remaining INTEGER DEFAULT 3;
        ALTER TABLE plays ADD COLUMN spread_line DOUBLE DEFAULT 1.5;
        ALTER TABLE plays ADD COLUMN posteam_score INTEGER DEFAULT 0;
        ALTER TABLE plays ADD COLUMN defteam_score INTEGER DEFAULT 3;
        ALTER TABLE plays ADD COLUMN wp DOUBLE DEFAULT 0.4;
        ALTER TABLE plays ADD COLUMN posteam_is_home BOOLEAN;
        UPDATE plays SET posteam_is_home = posteam = 'KC';
        CREATE TABLE games AS SELECT '2023_22_SF_KC' AS game_id, 'KC' AS home_team,
            'SF' AS away_team, 25 AS home_score, 22 AS away_score;
        """
    )
    yield connection
    connection.close()


def test_a_game_from_the_warehouse_is_its_scrimmage_plays_in_order_then_its_final(game_tables):
    *plays, final = replay.game_events(game_tables, "2023_22_SF_KC")
    assert [play.description for play in plays] == ["first", "second", "third"]
    assert [play.sequence for play in plays] == [0, 1, 2] and final.sequence == 3
    assert (final.home_score, final.away_score) == (25, 22)
    first, second, _ = plays
    # The home team's view: KC has the ball first, then SF (KC's spread and wp flip).
    assert (first.posteam_spread, first.nflfastr_home_wp) == (1.5, 0.4)
    assert (second.posteam_spread, second.nflfastr_home_wp) == (-1.5, pytest.approx(0.6))
    assert (second.home_score, second.away_score) == (3, 0)


def test_a_game_without_a_posted_result_ends_on_its_last_score(game_tables):
    game_tables.execute("UPDATE games SET home_score = NULL, away_score = NULL")
    final = replay.game_events(game_tables, "2023_22_SF_KC")[-1]
    assert (final.home_score, final.away_score) == (3, 0)


def test_an_unknown_game_is_an_error(game_tables):
    with pytest.raises(replay.GameNotFound):
        replay.game_events(game_tables, "1999_01_XXX_YYY")


def test_replay_is_paced_by_the_game_clock_and_capped():
    assert replay.pauses(GAME, speed=60) == [0.0, 0.5, replay.MAX_PAUSE, 0.0]
    assert replay.pauses(GAME, speed=0) == [0.0, 0.0, 0.0, 0.0]
    overtime = [_play(0, clock=2), _play(1, clock=600, qtr=5)]
    assert replay.pauses(overtime, speed=1) == [0.0, 0.0]


def test_replay_publishes_every_event_keyed_by_game():
    broker = MemoryBroker()
    waited = []
    assert replay.replay(broker.producer(), GAME, speed=60, sleep=waited.append) == 4
    assert waited == [0.5, replay.MAX_PAUSE]
    assert {message.key for message in broker.topics[PLAYS_TOPIC]} == {"2023_22_SF_KC"}


# --- scoring ------------------------------------------------------------------------------


def test_win_probability_is_reported_for_the_home_team():
    _, _, live = _pipeline(GAME, offense_wins(0.7))
    timeline = live.timeline("2023_22_SF_KC")
    assert [r.home_wp for r in timeline] == [0.7, pytest.approx(0.3), 0.7, 1.0]
    assert timeline[-1].final


def test_each_change_is_credited_to_the_play_that_caused_it():
    _, _, live = _pipeline(GAME, offense_wins(0.7))
    first, second, *_ = live.timeline("2023_22_SF_KC")
    assert (first.home_wp_change, first.description) == (0.0, "start of the game")
    assert (second.home_wp_change, second.description) == (pytest.approx(-0.4), "play 0")


def test_a_tie_ends_at_even():
    tie = [*GAME[:-1], GameFinal("2023_22_SF_KC", 3, "KC", "SF", home_score=20, away_score=20)]
    _, _, live = _pipeline(tie, offense_wins(0.6))
    assert live.timeline("2023_22_SF_KC")[-1].home_wp == 0.5


def test_a_bad_message_goes_to_the_dead_letter_topic_and_does_not_block_the_rest():
    broker = MemoryBroker()
    broker.producer().send(PLAYS_TOPIC, "2023_22_SF_KC", b"not json")
    _, stats, live = _pipeline(GAME, offense_wins(0.7), broker)
    assert (stats.scored, stats.dead_lettered, stats.games_finished) == (4, 1, 1)
    (letter,) = [json.loads(value) for value in broker.values(DEAD_LETTER_TOPIC)]
    assert letter["offset"] == 0 and letter["reason"].startswith("not JSON")
    assert len(live.timeline("2023_22_SF_KC")) == 4


class Crash(RuntimeError):
    pass


class CrashingProducer:
    """Delivers results until the Nth, then dies before it is delivered."""

    def __init__(self, broker: MemoryBroker, crash_on: int) -> None:
        self._inner = broker.producer()
        self._left = crash_on

    def send(self, topic, key, value):
        if topic == WP_TOPIC:
            self._left -= 1
            if self._left == 0:
                raise Crash
        self._inner.send(topic, key, value)

    def flush(self):
        self._inner.flush()


def test_a_crashed_scorer_rescored_the_play_it_was_on_and_the_feed_holds_it_once():
    broker = MemoryBroker()
    replay.replay(broker.producer(), GAME, speed=0)
    with pytest.raises(Crash):
        scorer.run(
            broker.consumer([PLAYS_TOPIC], scorer.GROUP),
            CrashingProducer(broker, crash_on=3),
            offense_wins(0.7),
        )
    assert broker.committed[(scorer.GROUP, PLAYS_TOPIC)] == 1  # plays 0 and 1 delivered

    # A replacement in the same group resumes at the uncommitted play...
    stats = scorer.run(
        broker.consumer([PLAYS_TOPIC], scorer.GROUP),
        broker.producer(),
        offense_wins(0.7),
        idle_seconds=0,
    )
    assert stats.scored == 2
    # ... and a play whose result was delivered but not committed would arrive twice;
    # the feed keeps one point per play either way.
    broker.producer().send(WP_TOPIC, "2023_22_SF_KC", broker.values(WP_TOPIC)[0])
    live = feed.watch(
        broker.consumer([WP_TOPIC], feed.GROUP),
        feed.Feed(),
        emit=lambda line: None,
        stop_after_finals=None,
        idle_seconds=0,
    )
    assert [r.sequence for r in live.timeline("2023_22_SF_KC")] == [0, 1, 2, 3]


# --- the feed -----------------------------------------------------------------------------


def test_the_feed_line_and_summary_read_like_a_broadcast():
    swings = [
        _play(0, clock=3600),
        _play(1, clock=60, qtr=4, home_score=16, away_score=19, score_differential=-3),
        _play(2, clock=10, qtr=4, home_score=16, away_score=19, score_differential=-3),
        GameFinal("2023_22_SF_KC", 3, "KC", "SF", home_score=25, away_score=22),
    ]
    chances = iter([0.6, 0.3, 0.7])
    model = scorer.Scorer(probability=lambda state: next(chances), model_version="test")
    _, _, live = _pipeline(swings, model)
    timeline = live.timeline("2023_22_SF_KC")
    assert feed.line(timeline[2]).startswith(
        "Q4  0:10  SF 19-16 KC  [##############......] KC  70.0% (+40.0)"
    )
    summary = feed.summarize(live, "2023_22_SF_KC")
    assert (summary.plays, summary.lead_changes) == (3, 2)
    assert summary.biggest_swing is timeline[2] and summary.biggest_swing.description == "play 1"
    assert "biggest swing: +40.0 points for KC, from play 1" in summary.render()


def test_win_probability_messages_round_trip():
    _, _, live = _pipeline(GAME, offense_wins(0.7))
    result = live.timeline("2023_22_SF_KC")[1]
    assert decode_win_probability(encode(result)) == result


# --- Kafka settings -------------------------------------------------------------------------


def test_kafka_settings_carry_the_delivery_guarantees():
    produce = producer_config("redpanda:9092")
    assert produce["enable.idempotence"] is True and produce["acks"] == "all"
    consume = consumer_config("redpanda:9092", scorer.GROUP)
    assert consume["enable.auto.commit"] is False
    assert consume["auto.offset.reset"] == "earliest"
