"""Each service imports only its own heavy dependencies.

This is what lets every service ship as its own image with its own pip extra: if the
gateway ever imported torch, its image would need torch. Checked in a fresh interpreter
per service, because this test process has already imported everything.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

HEAVY = ("duckdb", "torch", "psycopg", "sklearn", "polars")

ALLOWED = {
    "gateway": set(),
    "warehouse": {"duckdb"},
    "retrieval": {"psycopg"},
    "models": {"torch", "sklearn", "duckdb", "polars"},
}


@pytest.mark.parametrize("service", sorted(ALLOWED))
def test_service_imports_only_its_own_dependencies(service):
    probe = (
        f"import sys, fourthdown.services.{service}; "
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    loaded = {name for name in result.stdout.strip().split(",") if name}
    assert loaded <= ALLOWED[service], f"{service} pulled in {loaded - ALLOWED[service]}"
