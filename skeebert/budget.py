"""The brain's daily spending cap, and the per-user rate limit.

Budget
------
Every Anthropic response carries a ``usage`` block. Its four token counts
(uncached input, output, cache reads, cache writes) times the configured
prices (``config.Prices``) is what that call really cost; that number is added
to the current UTC day's row in the ``spend`` table, so a restart never resets
the day's total.

Before each call the brain asks for a ``Reservation`` sized to the most that
call could cost (its estimated prompt plus ``max_tokens`` of output). The check
"spent today + reserved by calls still in flight + this reservation <= cap"
runs under one lock, so several threads can never all slip past the cap at
once. When the call returns, the reservation is settled with the real cost;
if it failed, the reservation is released and nothing is recorded.

When a reservation is refused Skeebert "falls asleep" for the rest of the UTC
day: the brain answers with the local fallback and the message shows 💤. The
next UTC day has its own (empty) row, so it wakes up on its own.

Rate limit
----------
``RateLimiter`` is an in-memory sliding one-hour window per user pseudonym.
It resets on restart, which is fine: it exists to stop one person from
spending everyone's budget, not to be an exact quota.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from .config import LONG_CONTEXT_THRESHOLD_TOKENS, Prices
from .store import Store

Clock = Callable[[], datetime]
log = logging.getLogger(__name__)
WRITE_ATTEMPTS = 3
WRITE_RETRY_SECONDS = 0.05


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_day(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class UsageTokens:
    """The four billed token counts of one response."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @classmethod
    def from_usage(cls, usage) -> "UsageTokens":
        """Read an ``anthropic.types.Usage`` (or anything shaped like one). Missing counts are 0."""

        def get(name: str) -> int:
            v = getattr(usage, name, None)
            return int(v) if v else 0

        return cls(
            input_tokens=get("input_tokens"),
            output_tokens=get("output_tokens"),
            cache_read_tokens=get("cache_read_input_tokens"),
            cache_write_tokens=get("cache_creation_input_tokens"),
        )


def cost_usd(tokens: UsageTokens, prices: Prices) -> float:
    """What a call with these token counts costs, in USD."""
    long = tokens.prompt_tokens > LONG_CONTEXT_THRESHOLD_TOKENS
    p_in = prices.long_input if long else prices.input
    p_out = prices.long_output if long else prices.output
    p_cr = prices.long_cache_read if long else prices.cache_read
    p_cw = prices.long_cache_write if long else prices.cache_write
    return (
        tokens.input_tokens * p_in
        + tokens.output_tokens * p_out
        + tokens.cache_read_tokens * p_cr
        + tokens.cache_write_tokens * p_cw
    ) / 1_000_000


@dataclass
class Reservation:
    day: str
    amount: float
    open: bool = True


class Budget:
    def __init__(self, store: Store, daily_cap_usd: float, prices: Prices, clock: Clock = utc_now) -> None:
        if daily_cap_usd < 0:
            raise ValueError("daily cap must be >= 0")
        self.store = store
        self.cap = float(daily_cap_usd)
        self.prices = prices
        self.clock = clock
        self._lock = threading.Lock()
        self._pending: dict[str, float] = {}

    def today(self) -> str:
        return utc_day(self.clock())

    def spent_today(self) -> float:
        return self.store.spent_on(self.today())

    def remaining_today(self) -> float:
        with self._lock:
            day = self.today()
            return max(0.0, self.cap - self.store.spent_on(day) - self._pending.get(day, 0.0))

    def is_asleep(self) -> bool:
        """True when today's recorded spend has reached the cap."""
        return self.spent_today() >= self.cap

    def max_cost(self, prompt_tokens: int, max_output_tokens: int) -> float:
        """Worst-case cost of a call: the whole prompt uncached (or written to cache) plus full output."""
        worst_in = max(self.prices.input, self.prices.cache_write)
        return (prompt_tokens * worst_in + max_output_tokens * self.prices.output) / 1_000_000

    def reserve(self, amount: float) -> Reservation | None:
        """Hold ``amount`` of today's budget for a call about to be made, or None if it doesn't fit."""
        with self._lock:
            day = self.today()
            pending = self._pending.get(day, 0.0)
            if self.store.spent_on(day) + pending + amount > self.cap:
                return None
            self._pending[day] = pending + amount
            return Reservation(day, amount)

    def _close(self, reservation: Reservation) -> None:
        if not reservation.open:
            raise ValueError("reservation already settled or released")
        reservation.open = False
        left = self._pending.get(reservation.day, 0.0) - reservation.amount
        if left <= 1e-12:
            self._pending.pop(reservation.day, None)
        else:
            self._pending[reservation.day] = left

    def settle(self, reservation: Reservation, tokens: UsageTokens, *, usd: float | None = None) -> float:
        """Record what a finished call cost (on the reservation's day). Returns that cost.

        ``usd`` overrides the price computed from ``tokens``; it is used when a
        call timed out and may have been billed without us seeing its usage,
        so the whole reservation is charged.

        The ledger write is retried a few times. If it still fails, the cost
        is logged and stays *pending* (still counted against today's cap until
        restart) instead of raising, so the reply goes out and the cap stays
        honest.
        """
        cost = cost_usd(tokens, self.prices) if usd is None else float(usd)
        with self._lock:
            if not reservation.open:
                raise ValueError("reservation already settled or released")
            last_error: Exception | None = None
            for attempt in range(WRITE_ATTEMPTS):
                try:
                    self.store.add_spend(
                        reservation.day, cost,
                        input_tokens=tokens.input_tokens, output_tokens=tokens.output_tokens,
                        cache_read_tokens=tokens.cache_read_tokens, cache_write_tokens=tokens.cache_write_tokens,
                    )
                    self._close(reservation)
                    return cost
                except Exception as exc:  # sqlite busy/locked/disk errors
                    last_error = exc
                    time.sleep(WRITE_RETRY_SECONDS * (attempt + 1))
            # Could not record: keep the larger of reservation and real cost pending for the day.
            reservation.open = False
            extra = max(0.0, cost - reservation.amount)
            self._pending[reservation.day] = self._pending.get(reservation.day, 0.0) + extra
            log.error("could not record brain spend of $%.6f for %s (kept pending): %s", cost, reservation.day, last_error)
            return cost

    def release(self, reservation: Reservation) -> None:
        """Give a reservation back without recording spend (the call failed before any usage)."""
        with self._lock:
            self._close(reservation)


class RateLimiter:
    """At most ``per_hour`` acquisitions per key in any sliding 60-minute window."""

    WINDOW = 3600.0

    def __init__(self, per_hour: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_hour = int(per_hour)
        self.clock = clock
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def _trim(self, key: str, now: float) -> deque[float]:
        q = self._hits.setdefault(key, deque())
        while q and now - q[0] >= self.WINDOW:
            q.popleft()
        return q

    def try_acquire(self, key: str) -> bool:
        with self._lock:
            now = self.clock()
            q = self._trim(key, now)
            if len(q) >= self.per_hour:
                return False
            q.append(now)
            return True

    def refund(self, key: str) -> None:
        """Undo the most recent acquisition (used when the call never happened)."""
        with self._lock:
            q = self._hits.get(key)
            if q:
                q.pop()
