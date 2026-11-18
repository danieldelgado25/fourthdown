{{ config(materialized='view') }}
-- Players with a full name, for name resolution (fourthdown.rag.names) and generated SQL.
select
    player_id,
    full_name,
    position
from {{ ref('dim_player') }}
where full_name is not null
