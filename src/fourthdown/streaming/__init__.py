"""A game's play-by-play replayed through a broker into a live win-probability feed (phase 12).

::

    warehouse plays --replay--> nfl.plays --score--> nfl.win_probability --watch--> feed
                                    |
                                    +--(undecodable)--> nfl.plays.dlq

* ``replay`` publishes one game's scrimmage plays in order, keyed by game, paced by the
  game clock (``--speed 60``: an hour of game time in a minute), then a final event.
* ``score`` consumes ``nfl.plays``, evaluates each situation with the served
  win-probability model, and publishes the home team's win probability. Offsets are
  committed only after the result is delivered (at least once), and the feed keys results
  by play, so a redelivery overwrites rather than duplicates (effectively once).
* ``watch`` prints the feed as it arrives.

The broker is Redpanda (Kafka-compatible) in Docker Compose, or an in-process
``MemoryBroker`` with the same interface, which the tests and ``fourthdown stream demo``
use, so the whole path runs without a broker.
"""

from fourthdown.streaming.broker import KafkaBroker, MemoryBroker
from fourthdown.streaming.events import PLAYS_TOPIC, WP_TOPIC, GameFinal, PlayEvent, WinProbability

__all__ = [
    "PLAYS_TOPIC",
    "WP_TOPIC",
    "GameFinal",
    "KafkaBroker",
    "MemoryBroker",
    "PlayEvent",
    "WinProbability",
]
