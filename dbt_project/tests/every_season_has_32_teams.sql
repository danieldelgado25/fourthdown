-- Every season in the warehouse has all 32 clubs on offense.
select season, teams
from {{ ref('dim_season') }}
where teams <> 32
