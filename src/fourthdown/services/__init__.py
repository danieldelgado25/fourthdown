"""Independently deployable services.

=========  ====  ==========================  ===========================
service    port  owns                        extra (heavy dependencies)
=========  ====  ==========================  ===========================
gateway    8000  routing, public API, web    none
warehouse  8001  ``stats``, ``tendency``     ``warehouse`` (duckdb)
retrieval  8002  ``narrative``               ``retrieval`` (psycopg)
models     8003  ``advisor``                 ``models`` (torch, sklearn)
=========  ====  ==========================  ===========================

Tool services share one contract (`common.tool_service`); the gateway discovers their
tools over HTTP and imports none of their code. Each module runs with
``python -m fourthdown.services.<name>``. Nothing is imported here, so installing one
service's extra is enough to run it.
"""
