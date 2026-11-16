"""The idle watcher that deallocates the demo VM (fourthdown.serving.idle)."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import requests

from fourthdown.serving import idle
from fourthdown.serving.idle import Azure, AzureError, Settings, Watcher

COMPUTE = {"subscriptionId": "sub", "resourceGroupName": "rg", "name": "demo"}
RESOURCE = "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/demo"


@dataclass
class Reply:
    status_code: int
    body: dict[str, Any] = field(default_factory=dict)
    text: str = ""

    def json(self) -> dict[str, Any]:
        return self.body


@dataclass
class FakeAzure:
    """The metadata service and Resource Manager, recording what was asked."""

    deallocate_status: int = 202
    token_status: int = 200
    calls: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)

    def get(self, url: str, **kwargs: Any) -> Reply:
        self.calls.append(("GET", url, kwargs))
        if url.endswith("instance/compute"):
            return Reply(200, COMPUTE)
        if url.endswith("identity/oauth2/token"):
            return Reply(self.token_status, {"access_token": "t0ken"})
        raise AssertionError(url)

    def post(self, url: str, **kwargs: Any) -> Reply:
        self.calls.append(("POST", url, kwargs))
        return Reply(self.deallocate_status, text="no role assignment")


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def utmp(*kinds: int) -> bytes:
    return b"".join(struct.pack("<h", kind) + bytes(idle.UTMP_RECORD - 2) for kind in kinds)


@pytest.fixture()
def log(tmp_path) -> Path:
    return tmp_path / "queries.jsonl"


def settings(log: Path, tmp_path: Path, **overrides: Any) -> Settings:
    base = {
        "idle_minutes": 60.0,
        "activity": (log,),
        "utmp": tmp_path / "utmp",
    }
    return Settings(**{**base, **overrides})


def touch(path: Path, when: float) -> None:
    path.write_text("{}\n")
    os.utime(path, (when, when))


# --- the decision ----------------------------------------------------------------------


def test_login_sessions_counts_user_processes(tmp_path):
    path = tmp_path / "utmp"
    path.write_bytes(utmp(2, idle.USER_PROCESS, 8, idle.USER_PROCESS))  # boot, 2 users, dead
    assert idle.login_sessions(path) == 2
    assert idle.login_sessions(tmp_path / "missing") == 0


def test_last_activity_is_the_newest_existing_file(tmp_path, log):
    assert idle.last_activity([log]) is None
    touch(log, 500.0)
    assert idle.last_activity([log, tmp_path / "missing"]) == 500.0


@pytest.mark.parametrize(
    ("last", "sessions", "minutes", "expected"),
    [
        (None, 0, 60, True),  # nothing asked since the watcher started an hour ago
        (-30 * 60, 0, 60, False),  # a question 30 minutes ago
        (-61 * 60, 0, 60, True),  # a question 61 minutes ago
        (-61 * 60, 1, 60, False),  # but someone is logged in
        (None, 0, 0, False),  # watcher off
    ],
)
def test_due(last, sessions, minutes, expected, log, tmp_path):
    now = 10_000.0
    started = now - 60 * 60
    result = idle.due(
        settings(log, tmp_path, idle_minutes=minutes),
        now=now,
        started=started,
        last=None if last is None else now + last,
        sessions=sessions,
    )
    assert result is expected


# --- Azure -------------------------------------------------------------------------------


def test_deallocate_uses_the_managed_identity_on_this_vm():
    fake = FakeAzure()
    assert Azure(fake).deallocate() == RESOURCE
    method, url, kwargs = fake.calls[-1]
    assert method == "POST"
    assert url == f"{idle.ARM}{RESOURCE}/deallocate"
    assert kwargs["headers"] == {"Authorization": "Bearer t0ken"}
    assert all(call[2]["headers"] == {"Metadata": "true"} for call in fake.calls[:-1])


def test_a_missing_role_is_an_azure_error_that_says_so():
    with pytest.raises(AzureError, match="setup.sh"):
        Azure(FakeAzure(deallocate_status=403)).deallocate()


def test_no_managed_identity_is_an_azure_error():
    with pytest.raises(AzureError, match="answered 400"):
        Azure(FakeAzure(token_status=400)).deallocate()


def test_off_azure_the_metadata_service_is_unreachable():
    class Down:
        def get(self, url: str, **kwargs: Any) -> Any:
            raise requests.ConnectionError("no route to host")

        def post(self, url: str, **kwargs: Any) -> Any:
            raise AssertionError("never reached")

    with pytest.raises(AzureError, match="unreachable"):
        Azure(Down()).deallocate()


# --- the watcher -----------------------------------------------------------------------------


def test_watcher_deallocates_once_after_the_idle_window(log, tmp_path):
    clock, fake = Clock(), FakeAzure()
    watcher = Watcher(settings(log, tmp_path), Azure(fake), clock=clock)
    clock.now += 59 * 60
    assert not watcher.tick()
    clock.now += 2 * 60
    assert watcher.tick()
    clock.now += 60
    assert not watcher.tick()  # already asked; the VM is going down
    assert sum(call[0] == "POST" for call in fake.calls) == 1


def test_a_question_resets_the_window(log, tmp_path):
    clock, fake = Clock(), FakeAzure()
    watcher = Watcher(settings(log, tmp_path), Azure(fake), clock=clock)
    clock.now += 50 * 60
    touch(log, clock.now)
    clock.now += 50 * 60
    assert not watcher.tick()
    clock.now += 11 * 60
    assert watcher.tick()


def test_an_ssh_session_holds_the_vm_up(log, tmp_path):
    (tmp_path / "utmp").write_bytes(utmp(idle.USER_PROCESS))
    clock, fake = Clock(), FakeAzure()
    watcher = Watcher(settings(log, tmp_path), Azure(fake), clock=clock)
    clock.now += 5 * 60 * 60
    assert not watcher.tick()
    assert fake.calls == []


def test_log_action_never_calls_azure(log, tmp_path):
    clock, fake = Clock(), FakeAzure()
    watcher = Watcher(settings(log, tmp_path, action="log"), Azure(fake), clock=clock)
    clock.now += 2 * 60 * 60
    assert not watcher.tick()
    assert fake.calls == []


def test_a_failed_deallocate_waits_a_full_window_before_retrying(log, tmp_path):
    clock, fake = Clock(), FakeAzure(deallocate_status=403)
    watcher = Watcher(settings(log, tmp_path), Azure(fake), clock=clock)
    clock.now += 61 * 60
    assert not watcher.tick()
    clock.now += 60
    assert not watcher.tick()  # not every minute
    assert sum(call[0] == "POST" for call in fake.calls) == 1
    fake.deallocate_status = 202
    clock.now += 60 * 60
    assert watcher.tick()


def test_settings_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("FOURTHDOWN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FOURTHDOWN_IDLE_MINUTES", "45")
    monkeypatch.setenv("FOURTHDOWN_IDLE_ACTION", "log")
    loaded = Settings.from_env()
    assert loaded.idle_minutes == 45.0 and loaded.action == "log"
    assert loaded.activity == (tmp_path.resolve() / "feedback" / "queries.jsonl",)
