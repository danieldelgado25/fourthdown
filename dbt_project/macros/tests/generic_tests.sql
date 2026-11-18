{#- Generic data tests beyond dbt's four built-ins. Each returns the failing rows (or one
    row describing the failure), so `dbt test --store-failures` shows what broke. Written
    here rather than pulled from dbt_utils so a build never needs `dbt deps` (or network). -#}

{#- Every row satisfies a boolean SQL expression. -#}
{% test expression_is_true(model, expression, where=none) %}
select *
from {{ model }}
where not ({{ expression }})
{%- if where %} and ({{ where }}){% endif %}
{% endtest %}


{#- Non-null values of a column lie within [min_value, max_value]. -#}
{% test accepted_range(model, column_name, min_value, max_value) %}
select {{ column_name }}
from {{ model }}
where {{ column_name }} < {{ min_value }} or {{ column_name }} > {{ max_value }}
{% endtest %}


{#- The mean of an expression over the (optionally filtered) model lies within a band:
    a distribution check, for properties that hold league-wide but not row by row. -#}
{% test mean_between(model, expression, min_value, max_value, where=none) %}
with stats as (
    select avg({{ expression }}) as mean_value
    from {{ model }}
    {%- if where %} where {{ where }}{% endif %}
)
select mean_value
from stats
where mean_value is null or mean_value < {{ min_value }} or mean_value > {{ max_value }}
{% endtest %}


{#- The share of nulls in a column stays below a ceiling. -#}
{% test null_rate_below(model, column_name, max_rate, where=none) %}
with stats as (
    select avg(case when {{ column_name }} is null then 1.0 else 0.0 end) as null_rate
    from {{ model }}
    {%- if where %} where {{ where }}{% endif %}
)
select null_rate
from stats
where null_rate >= {{ max_rate }}
{% endtest %}


{#- No two rows share the combination of columns: the grain of an aggregate model. -#}
{% test unique_combination_of_columns(model, combination_of_columns) %}
select {{ combination_of_columns | join(', ') }}, count(*) as rows_with_key
from {{ model }}
group by {{ combination_of_columns | join(', ') }}
having count(*) > 1
{% endtest %}
