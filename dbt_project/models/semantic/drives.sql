-- One row per offensive drive.
select
    f.drive_id,
    any_value(f.game_id) as game_id,
    any_value(f.season) as season,
    any_value(g.season_type) as season_type,
    any_value(g.week) as week,
    any_value(f.posteam) as posteam,
    any_value(f.defteam) as defteam,
    any_value(f.fixed_drive) as drive_number,
    any_value(f.fixed_drive_result) as drive_result,
    min(f.qtr) as start_quarter,
    min_by(f.yardline_100, f.play_id) as start_yardline_100,
    max_by(f.yardline_100, f.play_id) as end_yardline_100,
    min_by(f.game_seconds_remaining, f.play_id) as end_game_seconds_remaining,
    count(*) as plays,
    count_if(f.is_designed_play) as designed_plays,
    sum(f.yards_gained) as yards_gained,
    sum(f.first_down::int) as first_downs,
    sum(f.epa) as epa,
    avg(f.epa) as epa_per_play,
    coalesce(any_value(f.fixed_drive_result) = 'Touchdown', false) as scored_touchdown
from {{ ref('fact_plays') }} as f
inner join {{ ref('dim_game') }} as g on f.game_id = g.game_id
where f.drive_id is not null and f.posteam is not null
group by f.drive_id
