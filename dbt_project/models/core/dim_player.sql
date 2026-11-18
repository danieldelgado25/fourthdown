-- One row per player id: everyone in nflverse's player table, plus anyone the play-by-play
-- credits as a passer, rusher, or receiver whom that table lacks, so every player key on
-- fact_plays resolves here (the relationships tests hold it to that).
with credited as (
    select passer_player_id as player_id, passer_player_name as player_name
    from {{ ref('stg_plays') }}
    where passer_player_id is not null
    union all
    select rusher_player_id, rusher_player_name
    from {{ ref('stg_plays') }}
    where rusher_player_id is not null
    union all
    select receiver_player_id, receiver_player_name
    from {{ ref('stg_plays') }}
    where receiver_player_id is not null
),

play_by_play as (
    select
        player_id,
        -- The abbreviated name the play-by-play uses most often for this id.
        mode(player_name) as short_name,
        count(*) as credited_plays
    from credited
    group by player_id
)

select
    coalesce(roster.player_id, play_by_play.player_id) as player_id,
    roster.full_name,
    roster.position,
    play_by_play.short_name,
    coalesce(play_by_play.credited_plays, 0) as credited_plays
from {{ ref('stg_players') }} as roster
full outer join play_by_play on roster.player_id = play_by_play.player_id
