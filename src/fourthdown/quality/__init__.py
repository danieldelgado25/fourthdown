"""Data quality: contracts on what comes in, and on what the models may learn from (phase 10).

* ``contracts``: Pandera schemas for the raw nflverse file (structure: required columns,
  numeric types, unique keys, the right season) and for each processed weekly partition
  (exact columns and types, value ranges, allowed labels, the partition's own season/week).
  The ETL runs both, so a bad file never becomes a partition.
* ``volume``: row counts. Plays per game and per week inside the bands every season since
  2009 falls in, and a re-delivered week within 5% of what was ingested before.
* ``leakage``: the feature contract. Every model feature is a declared pre-snap column,
  none is another model's output, and none predicts its target almost perfectly. Training
  refuses to run when it is broken.
* ``freshness``: during the season, every week the schedule shows over (with a grace
  period) has been ingested.
* ``alerts``: where failures go: the log, the Dagster UI, and an optional webhook.

Every check reports the same way: a ``DataQualityError`` naming what was checked and each
thing that failed, in words, with example values.
"""

from __future__ import annotations

from collections.abc import Sequence


class DataQualityError(ValueError):
    """Data failed a quality check; ``problems`` lists each failure in plain words."""

    def __init__(self, subject: str, problems: Sequence[str]) -> None:
        self.subject = subject
        self.problems = list(problems)
        count = len(self.problems)
        super().__init__(
            f"{subject} failed {count} data quality check{'s' if count != 1 else ''}:\n"
            + "\n".join(f"  - {problem}" for problem in self.problems)
        )
