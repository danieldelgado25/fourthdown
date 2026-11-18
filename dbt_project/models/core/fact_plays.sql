-- The fact table: one row per play (play_key), with its situation, outcome, and nflfastR
-- model outputs as measures, and keys to the dimensions: game_id -> dim_game,
-- season -> dim_season, posteam/defteam -> dim_team, and the passer, rusher, and
-- receiver ids -> dim_player. Game-level attributes live on dim_game, not here.
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
