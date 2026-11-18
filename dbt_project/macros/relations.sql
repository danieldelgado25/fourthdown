{#- On Snowflake, every ref() and source() renders without its database: `core.fact_plays`,
    not `FOURTHDOWN_BUILDING.core.fact_plays`.

    fourthdown.data.snowflake builds into a zero-copy clone of the live database and swaps
    the two only when every data test passed. A view keeps its SQL text through the swap,
    so a database name written into it would still point at the clone (dropped after the
    swap). Snowflake resolves a two-part name in a view against the view's own database,
    which is the live one once swapped in. Everything the project builds lives in one
    database, so nothing is lost. DuckDB names its catalog after the file and the build
    swaps files, so DuckDB keeps dbt's default. -#}

{% macro ref() %}
    {%- set version = kwargs.get('version') or kwargs.get('v') -%}
    {%- if (varargs | length) == 1 -%}
        {%- set relation = builtins.ref(varargs[0], version=version) -%}
    {%- else -%}
        {%- set relation = builtins.ref(varargs[0], varargs[1], version=version) -%}
    {%- endif -%}
    {%- do return(fourthdown_relation(relation)) -%}
{% endmacro %}


{% macro source(source_name, table_name) %}
    {%- do return(fourthdown_relation(builtins.source(source_name, table_name))) -%}
{% endmacro %}


{% macro fourthdown_relation(relation) %}
    {%- if target.type == 'snowflake' and relation is not none -%}
        {%- do return(relation.include(database=false)) -%}
    {%- endif -%}
    {%- do return(relation) -%}
{% endmacro %}
