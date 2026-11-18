-- The fact table: one row per play (play_key), with its situation, outcome, and nflfastR
-- model outputs as measures, and keys to the dimensions: game_id -> dim_game,
-- season -> dim_season, posteam/defteam -> dim_team, and the passer, rusher, and
-- receiver ids -> dim_player. Game-level attributes live on dim_game, not here.
--
-- Incremental: the orchestrator passes the weeks whose partitions are new or changed
-- (var `weeks`, keys like '2026-10'), and only those weeks' games are deleted and
-- reinserted. Overwriting by game rather than by play also drops a play nflverse deleted.
-- Without `weeks` (or on --full-refresh, or into an empty database) it builds everything.
{{
    config(
        materialized='incremental',
        incremental_strategy='delete+insert',
        unique_key='game_id',
        on_schema_change='fail',
    )
}}

{% set weeks = var('weeks', none) %}

select
    * exclude (
        season_type,
        week,
        game_date,
        home_team,
        away_team,
        home_score,
        away_score,
        result,
        total,
        spread_line,
        total_line,
        div_game,
        roof,
        surface,
        temp,
        wind,
        stadium,
        home_coach,
        away_coach
    )
from {{ ref('stg_plays') }}
{% if is_incremental() and weeks is not none %}
where cast(season as varchar) || '-' || lpad(cast(week as varchar), 2, '0') in (
    {%- for key in weeks %}'{{ key }}'{{ ',' if not loop.last }}{% endfor -%}
    {%- if not weeks %}null{% endif -%}
)
{% endif %}
