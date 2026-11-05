from __future__ import annotations

from fourthdown.services.guests import (
    BLOCKED_MESSAGE,
    DAY_SECONDS,
    GuestLimiter,
    GuestPolicy,
)


class Clock:
    def __init__(self) -> None:
        self.now = 10 * DAY_SECONDS

    def __call__(self) -> float:
        return self.now


def _limiter(clock: Clock, **limits: int) -> GuestLimiter:
    return GuestLimiter(GuestPolicy("guest", **limits), clock=clock)


def test_a_session_runs_out_but_a_new_one_starts_fresh():
    clock = Clock()
    guests = _limiter(clock, per_session=2)
    for _ in range(2):
        assert guests.admit("1.1.1.1", "s1") is None
        clock.now += 30
    assert "per session" in (guests.admit("1.1.1.1", "s1") or "")
    assert guests.admit("1.1.1.1", "s2") is None


def test_clearing_cookies_does_not_reset_an_address():
    clock = Clock()
    guests = _limiter(clock, per_ip_daily=3)
    for session in ("a", "b", "c"):
        assert guests.admit("1.1.1.1", session) is None
        clock.now += 30
    assert "a day" in (guests.admit("1.1.1.1", "d") or "")
    assert guests.admit("2.2.2.2", "e") is None


def test_all_guests_share_a_daily_cap_that_resets_at_midnight():
    clock = Clock()
    guests = _limiter(clock, daily=2)
    assert guests.admit("1.1.1.1", "a") is None
    assert guests.admit("2.2.2.2", "b") is None
    assert "today" in (guests.admit("3.3.3.3", "c") or "")
    clock.now += DAY_SECONDS
    assert guests.admit("3.3.3.3", "c") is None


def test_a_flooding_address_is_blocked_until_the_block_runs_out():
    clock = Clock()
    guests = _limiter(clock, burst=3, per_session=1)
    # Refused asks still count: hammering past a limit is what flooding looks like.
    replies = [guests.admit("6.6.6.6", "s") for _ in range(4)]
    assert replies[0] is None and replies[-1] == BLOCKED_MESSAGE
    assert guests.blocked("6.6.6.6") and not guests.blocked("1.1.1.1")
    clock.now += 24 * 3600
    assert not guests.blocked("6.6.6.6")


def test_a_steady_pace_is_never_blocked():
    clock = Clock()
    guests = _limiter(clock, burst=3)
    for _ in range(10):
        assert guests.admit("1.1.1.1", "s") is None
        clock.now += 21
    assert not guests.blocked("1.1.1.1")


def test_no_guest_user_means_no_policy(monkeypatch):
    monkeypatch.delenv("FOURTHDOWN_GUEST_USER", raising=False)
    assert GuestPolicy.from_env() is None
    monkeypatch.setenv("FOURTHDOWN_GUEST_USER", "guest")
    monkeypatch.setenv("FOURTHDOWN_GUEST_SESSION_LIMIT", "5")
    monkeypatch.setenv("FOURTHDOWN_GUEST_DAILY_LIMIT", "")
    policy = GuestPolicy.from_env()
    assert policy is not None and policy.per_session == 5
    assert policy.daily == GuestPolicy("guest").daily
