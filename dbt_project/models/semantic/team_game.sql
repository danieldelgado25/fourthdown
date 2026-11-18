-- One row per team per game, from the offense's side of the ball.
select
    f.game_id,
    f.posteam as team,
    any_value(f.defteam) as opponent,
    any_value(f.season) as season,
    any_value(g.season_type) as season_type,
    any_value(g.week) as week,
    any_value(g.playoff_round) as playoff_round,
    any_value(f.posteam_is_home) as is_home,
    count(*) as plays,
    count_if(f.is_designed_play) as designed_plays,
    avg(case when f.is_designed_play then f.is_pass_call::int end) as pass_rate,
    avg(case when f.is_designed_play and f.is_neutral_script then f.is_pass_call::int end)
        as neutral_pass_rate,
    avg(case when f.is_designed_play then f.epa end) as epa_per_play,
    avg(case when f.is_designed_play and f.pass_play then f.epa end) as pass_epa_per_play,
    avg(case when f.is_designed_play and f.rush_play then f.epa end) as rush_epa_per_play,
    avg(case when f.is_designed_play then f.success::int end) as success_rate,
    sum(case when f.is_designed_play then f.yards_gained end) as yards_gained,
    sum(f.touchdown::int) as touchdowns,
    sum(f.interception::int) as interceptions,
    sum(f.fumble_lost::int) as fumbles_lost,
    sum(f.third_down_converted::int) as third_down_conversions,
    count_if(f.down = 3 and f.is_designed_play) as third_down_attempts,
    sum(f.fourth_down_converted::int) as fourth_down_conversions,
    count_if(f.down = 4 and f.is_designed_play) as fourth_down_attempts
from {{ ref('fact_plays') }} as f
inner join {{ ref('dim_game') }} as g on f.game_id = g.game_id
where f.posteam is not null
group by f.game_id, f.posteam
