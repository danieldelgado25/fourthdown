-- dim_game keeps one value per game for each attribute nflverse repeats on every play.
-- That is lossless only if no game carries two values for one of them; nulls count as a
-- value, so a game half-filled with weather fails too.
{% set attributes = [
    'season', 'season_type', 'week', 'game_date', 'home_team', 'away_team', 'home_score',
    'away_score', 'result', 'total', 'spread_line', 'total_line', 'div_game', 'roof',
    'surface', 'temp', 'wind', 'stadium', 'home_coach', 'away_coach'
] %}
select game_id
from {{ ref('stg_plays') }}
group by game_id
having
{%- for attribute in attributes %}
    count(distinct coalesce(cast({{ attribute }} as varchar), '<null>')) > 1
    {%- if not loop.last %} or{% endif %}
{%- endfor %}
