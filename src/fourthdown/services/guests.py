"""Limits for the shared guest login, so a public demo cannot spend the LLM budget.

In production Caddy authenticates every request and tells the gateway who made it:
``X-Fourthdown-User`` (set by Caddy, replacing anything the client sent) and
``X-Forwarded-For`` (Caddy discards a client-supplied value and appends the real address).
The gateway's port is not published there, so both headers can only come from Caddy.
Requests from any other login -- the owner's, or local development with no Caddy at all --
are never limited.

Only ``POST /api/ask`` counts, since it is the endpoint that calls the LLM. A guest gets:

* ``per_session`` questions per browser session (a cookie the gateway sets);
* ``per_ip_daily`` questions per address per day, so clearing cookies does not reset it;
* a share of ``daily``, the cap across all guests, so rotating addresses cannot exhaust
  the provider's daily quota and lock the owner out too.

An address that asks more than ``burst`` times in a minute -- refused asks included, which
is what hammering past a limit looks like -- is blocked from the whole site for
``block_hours``. Counts live in memory: a gateway restart forgives everyone, and each
replica counts on its own.

Configuration: ``FOURTHDOWN_GUEST_USER`` (the guest's login; unset disables all of this),
``FOURTHDOWN_GUEST_SESSION_LIMIT``, ``FOURTHDOWN_GUEST_IP_DAILY_LIMIT``, and
``FOURTHDOWN_GUEST_DAILY_LIMIT``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

USER_HEADER = "X-Fourthdown-User"
SESSION_COOKIE = "fourthdown_guest"
BURST_WINDOW_SECONDS = 60.0
DAY_SECONDS = 86_400
BLOCKED_MESSAGE = "Too many requests from your address; it is blocked for now."


@dataclass(frozen=True)
class GuestPolicy:
    """Who the guest is and how much it may ask.

    Attributes:
        user: The guest's login name, as Caddy reports it.
        per_session: Questions per browser session.
        per_ip_daily: Questions per address per day (UTC).
        daily: Questions per day across every guest.
        burst: Asks per minute from one address before it is blocked.
        block_hours: How long a blocked address stays blocked.
    """

    user: str
    per_session: int = 15
    per_ip_daily: int = 40
    daily: int = 200
    burst: int = 8
    block_hours: float = 24.0

    @classmethod
    def from_env(cls) -> GuestPolicy | None:
        """The policy from ``FOURTHDOWN_GUEST_*``, or ``None`` when no guest is configured."""
        user = os.environ.get("FOURTHDOWN_GUEST_USER", "").strip()
        if not user:
            return None
        defaults = cls(user)
        return cls(
            user,
            per_session=int(
                os.environ.get("FOURTHDOWN_GUEST_SESSION_LIMIT") or defaults.per_session
            ),
            per_ip_daily=int(
                os.environ.get("FOURTHDOWN_GUEST_IP_DAILY_LIMIT") or defaults.per_ip_daily
            ),
            daily=int(os.environ.get("FOURTHDOWN_GUEST_DAILY_LIMIT") or defaults.daily),
        )


class GuestLimiter:
    """Thread-safe counters enforcing a `GuestPolicy`.

    Args:
        policy: The limits.
        clock: Seconds since the epoch; injectable for tests.
    """

    def __init__(self, policy: GuestPolicy, clock: Callable[[], float] = time.time) -> None:
        self.policy = policy
        self._clock = clock
        self._lock = threading.Lock()
        self._day = -1
        self._sessions: Counter[str] = Counter()
        self._addresses: Counter[str] = Counter()
        self._total = 0
        self._recent: dict[str, deque[float]] = {}
        self._blocked: dict[str, float] = {}

    def blocked(self, address: str) -> bool:
        """Whether ``address`` is currently blocked."""
        with self._lock:
            until = self._blocked.get(address)
            if until is not None and until <= self._clock():
                del self._blocked[address]
                return False
            return until is not None

    def admit(self, address: str, session: str) -> str | None:
        """Record one ask, or say why it is refused.

        Returns:
            ``None`` when the ask may proceed (it has been counted), otherwise a message
            for the guest.
        """
        policy = self.policy
        with self._lock:
            now = self._clock()
            self._roll_over(now)
            recent = self._recent.setdefault(address, deque())
            recent.append(now)
            while recent[0] <= now - BURST_WINDOW_SECONDS:
                recent.popleft()
            if len(recent) > policy.burst:
                self._blocked[address] = now + policy.block_hours * 3600
                del self._recent[address]
                LOGGER.warning(
                    "blocked guest address %s: %d asks in a minute", address, len(recent)
                )
                return BLOCKED_MESSAGE
            if self._total >= policy.daily:
                return "The guest demo has reached today's question limit; try again tomorrow."
            if self._addresses[address] >= policy.per_ip_daily:
                return f"Guests may ask {policy.per_ip_daily} questions a day; try again tomorrow."
            if self._sessions[session] >= policy.per_session:
                return f"Guests may ask {policy.per_session} questions per session."
            self._total += 1
            self._addresses[address] += 1
            self._sessions[session] += 1
            return None

    def _roll_over(self, now: float) -> None:
        """Start a fresh day's counts at UTC midnight; blocks run out on their own."""
        day = int(now // DAY_SECONDS)
        if day != self._day:
            self._day = day
            self._sessions.clear()
            self._addresses.clear()
            self._total = 0
        cutoff = now - BURST_WINDOW_SECONDS
        self._recent = {a: times for a, times in self._recent.items() if times[-1] > cutoff}
