-- One row per club, from the teams seed.
select
    team,
    team_name,
    conference,
    division
from {{ ref('teams') }}
