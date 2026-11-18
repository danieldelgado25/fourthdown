-- The play-by-play carries only abbreviated names ('J.Smith-Njigba'), many of them shared,
-- plus ids; this is where full names come from.
{#- fourthdown.data.modeling sets this false when raw/players.parquet is absent, so the
    warehouse still builds offline; dim_player then carries play-by-play names only. -#}
{% if env_var('FOURTHDOWN_PLAYERS_AVAILABLE', 'true') == 'true' %}
select
    gsis_id as player_id,
    display_name as full_name,
    position
from {{ source('nflverse', 'players') }}
where gsis_id is not null
{% else %}
-- No players file yet (an offline build): an empty table of the same shape.
select
    cast(null as varchar) as player_id,
    cast(null as varchar) as full_name,
    cast(null as varchar) as position
where false
{% endif %}
