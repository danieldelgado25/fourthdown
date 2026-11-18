-- dim_season's 17/18-week rule, which the playoff rounds are computed from, agrees with
-- the last regular-season week actually played. A future schedule change fails here
-- instead of mislabelling every playoff game.
select season, regular_season_weeks, last_regular_season_week
from {{ ref('dim_season') }}
where postseason_games > 0
  and last_regular_season_week <> regular_season_weeks
