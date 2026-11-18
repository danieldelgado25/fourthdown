-- Offensive tendencies per team and season (regular season and playoffs together), over
-- designed plays only: kneels, spikes, and special teams are not a coordinator's call.
-- The neutral_ columns further restrict to neutral game script (win probability 0.2-0.8,
-- first three quarters, outside the two-minute warning), which removes the trailing-team-
-- throws effect that a raw pass rate mistakes for philosophy. The dashboard's tendency
-- panel reads this table (fourthdown.services.warehouse.TENDENCY_SQL).
with designed as (
    select
        posteam as team,
        season,
        down,
        is_neutral_script,
        cast(is_pass_call as integer) as pass_call,
        epa,
        cast(success as integer) as success,
        pass_oe
    from {{ ref('fact_plays') }}
    where posteam is not null and is_designed_play
),

fourth_downs as (
    select
        posteam as team,
        season,
        count_if(is_designed_play) as fourth_down_go_attempts,
        count_if(is_designed_play or punt_attempt or field_goal_attempt) as fourth_down_decisions
    from {{ ref('fact_plays') }}
    where posteam is not null and down = 4
    group by posteam, season
)

select
    designed.team,
    designed.season,
    count(*) as designed_plays,
    avg(pass_call) as pass_rate,
    avg(epa) as epa_per_play,
    avg(success) as success_rate,
    -- nflfastR's pass_oe is in percentage points.
    avg(pass_oe) as pass_rate_over_expected,
    count_if(is_neutral_script) as neutral_plays,
    avg(case when is_neutral_script then pass_call end) as neutral_pass_rate,
    avg(case when is_neutral_script and down in (1, 2) then pass_call end)
        as neutral_early_down_pass_rate,
    avg(case when is_neutral_script then epa end) as neutral_epa_per_play,
    any_value(fourth_downs.fourth_down_go_attempts) as fourth_down_go_attempts,
    any_value(fourth_downs.fourth_down_go_attempts)
        / nullif(any_value(fourth_downs.fourth_down_decisions), 0) as fourth_down_go_rate
from designed
left join fourth_downs
    on designed.team = fourth_downs.team and designed.season = fourth_downs.season
group by designed.team, designed.season
