-- One row per season. regular_season_weeks is the rule the playoff rounds hang off: the
-- season grew from 16 to 17 games in 2021, moving the last regular-season week from 17 to
-- 18 (the test regular_season_weeks_match_the_schedule holds the rule to the data).
select
    season,
    case when season >= 2021 then 18 else 17 end as regular_season_weeks,
    count(distinct case when season_type = 'REG' then game_id end) as regular_season_games,
    count(distinct case when season_type = 'POST' then game_id end) as postseason_games,
    count(distinct posteam) as teams,
    max(case when season_type = 'REG' then week end) as last_regular_season_week,
    min(game_date) as first_game_date,
    max(game_date) as last_game_date,
    count(*) as plays
from {{ ref('stg_plays') }}
group by season
