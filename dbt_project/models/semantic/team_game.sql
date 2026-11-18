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
    count(*) filter (where f.is_designed_play) as designed_plays,
    avg(f.is_pass_call::int) filter (where f.is_designed_play) as pass_rate,
    avg(f.is_pass_call::int) filter (where f.is_designed_play and f.is_neutral_script)
        as neutral_pass_rate,
    avg(f.epa) filter (where f.is_designed_play) as epa_per_play,
    avg(f.epa) filter (where f.is_designed_play and f.pass_play) as pass_epa_per_play,
    avg(f.epa) filter (where f.is_designed_play and f.rush_play) as rush_epa_per_play,
    avg(f.success::int) filter (where f.is_designed_play) as success_rate,
    sum(f.yards_gained) filter (where f.is_designed_play) as yards_gained,
    sum(f.touchdown::int) as touchdowns,
    sum(f.interception::int) as interceptions,
    sum(f.fumble_lost::int) as fumbles_lost,
    sum(f.third_down_converted::int) as third_down_conversions,
    count(*) filter (where f.down = 3 and f.is_designed_play) as third_down_attempts,
    sum(f.fourth_down_converted::int) as fourth_down_conversions,
    count(*) filter (where f.down = 4 and f.is_designed_play) as fourth_down_attempts
from {{ ref('fact_plays') }} as f
inner join {{ ref('dim_game') }} as g on f.game_id = g.game_id
where f.posteam is not null
group by f.game_id, f.posteam
