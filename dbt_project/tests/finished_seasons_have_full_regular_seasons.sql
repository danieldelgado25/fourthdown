-- A season that reached the playoffs has a full regular season behind it: 256 games
-- through 2020, 272 since. A season still being played is exempt until its postseason.
select season, regular_season_games
from {{ ref('dim_season') }}
where postseason_games > 0
  and regular_season_games < 256
