{#- Use a model's schema as written (core, marts) instead of dbt's default <target>_<custom>,
    so the warehouse has the same schema names on every target. -#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
