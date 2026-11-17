"""Deallocate the demo VM when nobody has used it for a while.

The demo runs on an Azure VM that bills by the hour while it is allocated. Shutting the OS
down is not enough: a VM stopped from inside is "Stopped", still allocated, and still
billed for compute. Only a deallocate, an Azure API call, stops the meter. So this
watcher, run as a service beside the app (``docker-compose.prod.yml``), asks Azure to
deallocate its own VM once

- no question has reached the gateway for ``FOURTHDOWN_IDLE_MINUTES`` (default 60;
  the gateway's query log, `fourthdown.feedback`, is the activity signal), and
- nobody is logged in over SSH (the host's ``utmp``, mounted read-only), so a working
  session is never cut off.

The clock starts when the watcher starts, so a freshly started VM always gets the full
idle window. The call authenticates as the VM's managed identity through the instance
metadata service, so no credential lives on the machine; ``deploy/azure/setup.sh``
assigns the identity and a role scoped to this one VM. Without it, the deallocate fails
and is logged, and nothing else happens. ``FOURTHDOWN_IDLE_ACTION=log`` only reports
what it would do; ``FOURTHDOWN_IDLE_MINUTES=0`` turns the watcher off.
"""

from __future__ import annotations

import logging
import os
import signal
import struct
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from fourthdown.config import data_paths
from fourthdown.feedback import QUERIES_FILENAME

LOGGER = logging.getLogger(__name__)

IMDS = "http://169.254.169.254/metadata"
ARM = "https://management.azure.com"
COMPUTE_API = "2023-03-01"
TIMEOUT = 10.0

# glibc's struct utmp on Linux x86-64: 384 bytes, beginning with a short ut_type.
UTMP_RECORD = 384
USER_PROCESS = 7


HTTP = Any
"""The transport: the `requests` module in production (only ``get`` and ``post``, as in
`fourthdown.services.gateway`); tests pass a stand-in."""


@dataclass(frozen=True)
class Settings:
    idle_minutes: float = 60.0
    action: str = "deallocate"
    interval_seconds: float = 60.0
    activity: tuple[Path, ...] = ()
    utmp: Path = Path("/host/utmp")

    @classmethod
    def from_env(cls) -> Settings:
        feedback = data_paths().feedback
        return cls(
            idle_minutes=float(os.environ.get("FOURTHDOWN_IDLE_MINUTES", "60")),
            action=os.environ.get("FOURTHDOWN_IDLE_ACTION", "deallocate"),
            interval_seconds=float(os.environ.get("FOURTHDOWN_IDLE_INTERVAL", "60")),
            activity=(feedback / QUERIES_FILENAME,),
            utmp=Path(os.environ.get("FOURTHDOWN_UTMP", "/host/utmp")),
        )


def last_activity(paths: Sequence[Path]) -> float | None:
    """The newest modification time among ``paths`` that exist (epoch seconds)."""
    times = [path.stat().st_mtime for path in paths if path.is_file()]
    return max(times) if times else None


def login_sessions(utmp: Path) -> int:
    """How many users are logged in, from a ``utmp`` file; 0 when it cannot be read."""
    try:
        data = utmp.read_bytes()
    except OSError:
        return 0
    count = 0
    for offset in range(0, len(data) - UTMP_RECORD + 1, UTMP_RECORD):
        (kind,) = struct.unpack_from("<h", data, offset)
        if kind == USER_PROCESS:
            count += 1
    return count


def idle_seconds(now: float, started: float, last: float | None) -> float:
    """Seconds since the last question, or since the watcher started if that is later."""
    return now - max(started, last or 0.0)


def due(
    settings: Settings, *, now: float, started: float, last: float | None, sessions: int
) -> bool:
    """Whether the VM has been idle long enough, with nobody logged in, to deallocate."""
    if settings.idle_minutes <= 0 or sessions > 0:
        return False
    return idle_seconds(now, started, last) >= settings.idle_minutes * 60


class AzureError(RuntimeError):
    """The metadata service or Azure Resource Manager refused or could not be reached."""


class Azure:
    """This VM, as Azure Resource Manager sees it, via its managed identity."""

    def __init__(self, http: HTTP = requests) -> None:
        self._http = http

    def _metadata(self, path: str, **params: str) -> dict[str, Any]:
        try:
            response = self._http.get(
                f"{IMDS}/{path}", headers={"Metadata": "true"}, params=params, timeout=TIMEOUT
            )
        except requests.RequestException as error:
            raise AzureError(f"instance metadata unreachable: {error}") from error
        if response.status_code != 200:
            raise AzureError(f"instance metadata {path} answered {response.status_code}")
        body: dict[str, Any] = response.json()
        return body

    def resource_id(self) -> str:
        compute = self._metadata("instance/compute", **{"api-version": "2021-02-01"})
        return (
            f"/subscriptions/{compute['subscriptionId']}/resourceGroups/"
            f"{compute['resourceGroupName']}/providers/Microsoft.Compute/virtualMachines/"
            f"{compute['name']}"
        )

    def token(self) -> str:
        """A management-plane token for the VM's managed identity."""
        body = self._metadata(
            "identity/oauth2/token", **{"api-version": "2018-02-01", "resource": f"{ARM}/"}
        )
        return str(body["access_token"])

    def deallocate(self) -> str:
        """Ask Azure to deallocate this VM; returns its resource id once accepted."""
        resource = self.resource_id()
        token = self.token()
        try:
            response = self._http.post(
                f"{ARM}{resource}/deallocate",
                params={"api-version": COMPUTE_API},
                headers={"Authorization": f"Bearer {token}"},
                timeout=TIMEOUT,
            )
        except requests.RequestException as error:
            raise AzureError(f"Azure Resource Manager unreachable: {error}") from error
        if response.status_code not in (200, 202):
            raise AzureError(
                f"deallocate answered {response.status_code}: {response.text[:200]} "
                "(has deploy/azure/setup.sh given the VM its identity and role?)"
            )
        return resource


@dataclass
class Watcher:
    """One check per `tick`; `run` loops forever."""

    settings: Settings
    azure: Azure
    clock: Callable[[], float] = time.time
    started: float = 0.0
    requested: bool = False
    reported: bool = False

    def __post_init__(self) -> None:
        self.started = self.started or self.clock()

    def tick(self) -> bool:
        """Check once; True when this tick asked Azure to deallocate."""
        now = self.clock()
        last = last_activity(self.settings.activity)
        sessions = login_sessions(self.settings.utmp)
        if not due(self.settings, now=now, started=self.started, last=last, sessions=sessions):
            self.reported = False
            return False
        idle = idle_seconds(now, self.started, last) / 60
        if self.settings.action != "deallocate":
            if not self.reported:
                LOGGER.info("idle %.0f minutes; would deallocate (action=log)", idle)
                self.reported = True
            return False
        if self.requested:
            return False
        LOGGER.warning("idle %.0f minutes with nobody logged in; deallocating", idle)
        try:
            resource = self.azure.deallocate()
        except AzureError as error:
            LOGGER.error("could not deallocate: %s", error)
            # Retry on the next idle window rather than every minute.
            self.started = now
            return False
        self.requested = True
        LOGGER.warning("deallocate accepted for %s", resource)
        return True

    def run(self) -> None:
        if self.settings.idle_minutes <= 0:
            LOGGER.info("FOURTHDOWN_IDLE_MINUTES is 0; the idle watcher is off")
            while True:
                time.sleep(3600)
        LOGGER.info(
            "deallocating after %.0f idle minutes (action=%s), watching %s",
            self.settings.idle_minutes,
            self.settings.action,
            ", ".join(str(path) for path in self.settings.activity),
        )
        while True:
            self.tick()
            time.sleep(self.settings.interval_seconds)


def _exit_on_sigterm(signum: int, frame: object) -> None:
    raise SystemExit(0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # As a container's PID 1, Python ignores SIGTERM unless it handles it.
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    Watcher(Settings.from_env(), Azure()).run()


if __name__ == "__main__":
    main()
