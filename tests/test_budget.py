import threading
from datetime import datetime, timedelta, timezone

import pytest

from skeebert.budget import Budget, RateLimiter, UsageTokens, cost_usd
from skeebert.config import Prices
from skeebert.store import Store


# Same rates at any prompt length, so the big round token counts below price simply.
FLAT = Prices(long_input=0.10, long_output=0.50, long_cache_read=0.01, long_cache_write=0.125)


class Clock:
    def __init__(self, when):
        self.now = when

    def __call__(self):
        return self.now


def test_cost_from_usage():
    p = Prices()
    t = UsageTokens(input_tokens=20_000, output_tokens=20_000, cache_read_tokens=20_000, cache_write_tokens=20_000)
    assert cost_usd(t, p) == pytest.approx(0.02 * (0.10 + 0.50 + 0.01 + 0.125))
    small = UsageTokens(input_tokens=200, output_tokens=50, cache_read_tokens=3000)
    assert cost_usd(small, p) == pytest.approx((200 * 0.10 + 50 * 0.50 + 3000 * 0.01) / 1e6)


def test_long_prompt_uses_long_rate_card():
    p = Prices()
    t = UsageTokens(input_tokens=100_001, output_tokens=10)
    assert cost_usd(t, p) == pytest.approx((100_001 * 0.50 + 10 * 2.50) / 1e6)


def test_usage_tokens_from_sdk_usage():
    from anthropic.types import Usage

    u = Usage(input_tokens=5, output_tokens=6, cache_read_input_tokens=7, cache_creation_input_tokens=None)
    assert UsageTokens.from_usage(u) == UsageTokens(5, 6, 7, 0)


def test_cap_makes_it_sleep_and_persists(tmp_path):
    clock = Clock(datetime(2026, 10, 8, 23, 0, tzinfo=timezone.utc))
    store = Store(tmp_path / "x.db")
    b = Budget(store, 1.0, FLAT, clock)
    r = b.reserve(0.4)
    assert r is not None
    b.settle(r, UsageTokens(input_tokens=9_000_000))  # $0.90 real cost
    assert b.spent_today() == pytest.approx(0.9)
    assert not b.is_asleep()
    assert b.reserve(0.2) is None  # would exceed the cap
    r2 = b.reserve(0.1)
    b.settle(r2, UsageTokens(output_tokens=200_000))  # $0.10
    assert b.is_asleep()
    # a new instance (restart) sees the same day's spend
    b2 = Budget(Store(tmp_path / "x.db"), 1.0, FLAT, clock)
    assert b2.is_asleep()
    assert b2.reserve(0.0001) is None
    # next UTC day: awake again
    clock.now = clock.now + timedelta(hours=1, minutes=1)
    assert b2.today() == "2026-10-09"
    assert not b2.is_asleep()
    assert b2.reserve(0.5) is not None


def test_release_records_nothing(tmp_path):
    b = Budget(Store(tmp_path / "x.db"), 1.0, FLAT)
    r = b.reserve(0.9)
    assert b.reserve(0.2) is None  # pending counts against the cap
    b.release(r)
    assert b.spent_today() == 0.0
    assert b.reserve(0.9) is not None
    with pytest.raises(ValueError):
        b.release(r)  # double release


def test_reservation_settles_on_its_own_day(tmp_path):
    clock = Clock(datetime(2026, 10, 8, 23, 59, 59, tzinfo=timezone.utc))
    store = Store(tmp_path / "x.db")
    b = Budget(store, 1.0, FLAT, clock)
    r = b.reserve(0.01)
    clock.now += timedelta(seconds=5)
    b.settle(r, UsageTokens(output_tokens=1000))
    assert store.spent_on("2026-10-08") > 0
    assert store.spent_on("2026-10-09") == 0


def test_concurrent_reservations_never_exceed_cap(tmp_path):
    b = Budget(Store(tmp_path / "x.db"), 1.0, FLAT)
    granted = []
    lock = threading.Lock()
    barrier = threading.Barrier(16)

    def worker():
        barrier.wait()
        r = b.reserve(0.3)
        if r is not None:
            with lock:
                granted.append(r)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(granted) == 3  # 3 x 0.3 <= 1.0 < 4 x 0.3
    for r in granted:
        b.settle(r, UsageTokens(input_tokens=3_000_000))  # $0.30 each
    assert b.spent_today() == pytest.approx(0.9)


def test_max_cost_is_an_upper_bound():
    b = Budget.__new__(Budget)
    b.prices = Prices()
    worst = Budget.max_cost(b, 5000, 1024)
    actual = cost_usd(UsageTokens(cache_write_tokens=5000, output_tokens=1024), Prices())
    assert worst >= actual


def test_rate_limiter_sliding_window():
    t = [0.0]
    rl = RateLimiter(3, clock=lambda: t[0])
    assert all(rl.try_acquire("u") for _ in range(3))
    assert not rl.try_acquire("u")
    assert rl.try_acquire("other")  # per key
    t[0] = 3599.0
    assert not rl.try_acquire("u")
    t[0] = 3600.0
    assert rl.try_acquire("u")
    rl.refund("u")
    assert rl.try_acquire("u")
    assert not hasattr(rl, "forget")  # nothing may reset a person's window early


class FlakyStore:
    """Wraps a Store; ``add_spend`` fails ``failures`` times first."""

    def __init__(self, store, failures):
        self.store, self.failures = store, failures

    def __getattr__(self, name):
        return getattr(self.store, name)

    def add_spend(self, *a, **k):
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("database is locked")
        return self.store.add_spend(*a, **k)


def test_settle_retries_ledger_write(tmp_path):
    store = Store(tmp_path / "x.db")
    b = Budget(FlakyStore(store, 2), 1.0, FLAT)
    r = b.reserve(0.1)
    b.settle(r, UsageTokens(output_tokens=100_000))  # $0.05
    assert store.spent_on(b.today()) == pytest.approx(0.05)
    assert b.remaining_today() == pytest.approx(0.95)


def test_settle_failure_keeps_cost_pending_and_does_not_raise(tmp_path):
    store = Store(tmp_path / "x.db")
    b = Budget(FlakyStore(store, 99), 1.0, FLAT)
    r = b.reserve(0.1)
    cost = b.settle(r, UsageTokens(output_tokens=400_000))  # $0.20, more than reserved
    assert cost == pytest.approx(0.20)
    assert store.spent_on(b.today()) == 0.0
    assert b.remaining_today() == pytest.approx(0.80)  # still counted against today's cap
    assert b.reserve(0.85) is None


def test_settle_with_explicit_usd(tmp_path):
    b = Budget(Store(tmp_path / "x.db"), 1.0, FLAT)
    r = b.reserve(0.3)
    assert b.settle(r, UsageTokens(), usd=r.amount) == pytest.approx(0.3)
    assert b.spent_today() == pytest.approx(0.3)
