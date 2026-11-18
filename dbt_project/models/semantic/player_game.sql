-- One row per player, role, and game. Box-score conventions: two-point tries are not
-- plays, sack yardage is charged to the team rather than the passer, and a touchdown counts
-- for a role only when that role scored it (`touchdown` alone also fires on the defence's
-- pick-six or fumble return). A receiver's `plays` are targets, incompletions included;
-- `completions` are the catches.
with plays as (
    select
        f.*,
        g.season_type,
        g.week,
        g.playoff_round
    from {{ ref('fact_plays') }} as f
    inner join {{ ref('dim_game') }} as g on f.game_id = g.game_id
    where not coalesce(f.two_point_attempt, false)
),

roles as (
    select
        game_id, season, season_type, week, playoff_round,
        posteam as team, 'passer' as role,
        passer_player_id as player_id, passer_player_name as player_name,
        count(*) as plays,
        coalesce(sum(case when not coalesce(sack, false) then yards_gained end), 0) as yards,
        sum(coalesce(pass_touchdown, false)::int) as touchdowns,
        sum(coalesce(complete_pass, false)::int) as completions,
        avg(epa) as epa_per_play,
        avg(cpoe) as cpoe
    from plays
    where passer_player_id is not null
    group by all
    union all
    select
        game_id, season, season_type, week, playoff_round,
        posteam as team, 'rusher' as role,
        rusher_player_id as player_id, rusher_player_name as player_name,
        count(*) as plays,
        sum(yards_gained) as yards,
        sum(coalesce(rush_touchdown, false)::int) as touchdowns,
        null as completions,
        avg(epa) as epa_per_play,
        null as cpoe
    from plays
    where rusher_player_id is not null
    group by all
    union all
    select
        game_id, season, season_type, week, playoff_round,
        posteam as team, 'receiver' as role,
        receiver_player_id as player_id, receiver_player_name as player_name,
        count(*) as plays,
        sum(yards_gained) as yards,
        sum(coalesce(pass_touchdown, false)::int) as touchdowns,
        sum(coalesce(complete_pass, false)::int) as completions,
        avg(epa) as epa_per_play,
        null as cpoe
    from plays
    where receiver_player_id is not null
    group by all
)

select
    roles.*,
    dim_player.full_name
from roles
left join {{ ref('dim_player') }} as dim_player on roles.player_id = dim_player.player_id
