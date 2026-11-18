-- One row per play, as the ETL wrote it, plus a single-column key for the play's grain.
select
    game_id || '_' || cast(play_id as varchar) as play_key,
    *
from {{ source('nflverse', 'processed_plays') }}
