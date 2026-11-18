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

name_uses as (
    select player_id, player_name, count(*) as uses
    from credited
    where player_name is not null
    group by player_id, player_name
),

short_names as (
    -- The abbreviated name the play-by-play uses most often for this id. A tie goes to the
    -- alphabetically first spelling, so every engine and every build order picks the same
    -- one (mode() leaves ties to the engine).
    select player_id, player_name as short_name
    from name_uses
    qualify row_number() over (partition by player_id order by uses desc, player_name) = 1
),

play_by_play as (
    select
        credited.player_id,
        any_value(short_names.short_name) as short_name,
        count(*) as credited_plays
    from credited
    left join short_names on credited.player_id = short_names.player_id
    group by credited.player_id
)

select
    coalesce(roster.player_id, play_by_play.player_id) as player_id,
    roster.full_name,
    roster.position,
    play_by_play.short_name,
    coalesce(play_by_play.credited_plays, 0) as credited_plays
from {{ ref('stg_players') }} as roster
full outer join play_by_play on roster.player_id = play_by_play.player_id
