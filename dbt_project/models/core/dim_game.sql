-- One row per game: everything nflverse repeats on each of a game's ~170 plays (teams,
-- score, betting lines, weather, venue, coaches), stored once. The test
-- game_attributes_are_constant_within_a_game proves the split from fact_plays is lossless.
with games as (
    select
        game_id,
        any_value(season) as season,
        any_value(season_type) as season_type,
        any_value(week) as week,
        any_value(game_date) as game_date,
        any_value(home_team) as home_team,
        any_value(away_team) as away_team,
        any_value(home_score) as home_score,
        any_value(away_score) as away_score,
        any_value(result) as home_margin,
        any_value(total) as total_points,
        any_value(spread_line) as spread_line,
        any_value(total_line) as total_line,
        any_value(div_game) as div_game,
        any_value(roof) as roof,
        any_value(surface) as surface,
        any_value(temp) as temp,
        any_value(wind) as wind,
        any_value(stadium) as stadium,
        any_value(home_coach) as home_coach,
        any_value(away_coach) as away_coach
    from {{ ref('stg_plays') }}
    group by game_id
)

select
    games.game_id,
    games.season,
    games.season_type,
    games.week,
    -- A round is the week's offset from the last regular-season week, not a fixed week.
    case when games.season_type = 'POST' then
        case games.week - seasons.regular_season_weeks
            when 1 then 'wild_card'
            when 2 then 'divisional'
            when 3 then 'conference'
            when 4 then 'super_bowl'
        end
    end as playoff_round,
    games.game_date,
    games.home_team,
    games.away_team,
    games.home_score,
    games.away_score,
    games.home_margin,
    games.total_points,
    games.spread_line,
    games.total_line,
    games.div_game,
    games.roof,
    games.surface,
    games.temp,
    games.wind,
    games.stadium,
    games.home_coach,
    games.away_coach
from games
inner join {{ ref('dim_season') }} as seasons on games.season = seasons.season
