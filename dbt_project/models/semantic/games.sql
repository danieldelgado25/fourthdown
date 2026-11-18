-- One row per game: dim_game plus its play count.
with play_counts as (
    select game_id, count(*) as plays
    from {{ ref('fact_plays') }}
    group by game_id
)

select
    g.*,
    play_counts.plays
from {{ ref('dim_game') }} as g
inner join play_counts on g.game_id = play_counts.game_id
