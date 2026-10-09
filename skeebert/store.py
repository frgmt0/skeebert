"""Everything Skeebert keeps, in one SQLite file (``data/skeebert.db``).

Tables
------
``users``      one row per pseudonym: whether we keep their stuff, whether the
               one-line first-time notice was shown, and their guess count
               (for the two one-off tips).
``exchanges``  something addressed to Skeebert and what it answered: the
               message text, the intent the brain chose, the exact glyph shown
               (stroke JSON, so training re-renders the same picture), the
               model version, which brain answered and whether it was asleep.
``guesses``    one human reading of one exchange's glyph, with its score. At
               most one per (exchange, user); deleting an exchange cascades.
``spend``      the brain's real API spend per UTC day (from response usage).
``state``      small global values, e.g. Skeebert's current mood.

What is never here: raw Discord ids (only salted hashes, see ``identity``),
server/channel names, avatars, or messages not addressed to Skeebert. Rows of a
person who pressed "Stop keeping" are not written at all (the service enforces
this before calling the store).

Thread safety: one connection shared by every thread, every statement under a
re-entrant lock. Migrations are numbered and applied in order on open, tracked
by ``PRAGMA user_version``; never edit a released migration, only add one.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .types import Glyph, Intent, TrainingExample

MIGRATIONS: tuple[str, ...] = (
    # 1: initial schema
    """
    CREATE TABLE users (
        user_hash     TEXT PRIMARY KEY,
        keeping       INTEGER NOT NULL DEFAULT 1,
        onboarded_at  TEXT,
        guesses_made  INTEGER NOT NULL DEFAULT 0,
        created_at    TEXT NOT NULL
    );
    CREATE TABLE exchanges (
        id             TEXT PRIMARY KEY,
        user_hash      TEXT NOT NULL,
        source         TEXT NOT NULL,
        message_text   TEXT NOT NULL,
        intent         TEXT NOT NULL,
        glyph          TEXT NOT NULL,
        model_version  TEXT NOT NULL,
        brain_source   TEXT NOT NULL,
        asleep         INTEGER NOT NULL,
        mood           TEXT,
        notice_shown   INTEGER NOT NULL DEFAULT 0,
        message_ref    TEXT,
        created_at     TEXT NOT NULL
    );
    CREATE INDEX exchanges_user ON exchanges(user_hash);
    CREATE INDEX exchanges_version_intent ON exchanges(model_version, intent);
    CREATE UNIQUE INDEX exchanges_message_ref ON exchanges(message_ref) WHERE message_ref IS NOT NULL;
    CREATE TABLE guesses (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        exchange_id  TEXT NOT NULL REFERENCES exchanges(id) ON DELETE CASCADE,
        user_hash    TEXT NOT NULL,
        guess_text   TEXT NOT NULL,
        score        REAL NOT NULL,
        via          TEXT NOT NULL,
        created_at   TEXT NOT NULL,
        UNIQUE (exchange_id, user_hash)
    );
    CREATE INDEX guesses_user ON guesses(user_hash);
    CREATE TABLE spend (
        day                 TEXT PRIMARY KEY,
        usd                 REAL NOT NULL DEFAULT 0,
        calls               INTEGER NOT NULL DEFAULT 0,
        input_tokens        INTEGER NOT NULL DEFAULT 0,
        output_tokens       INTEGER NOT NULL DEFAULT 0,
        cache_read_tokens   INTEGER NOT NULL DEFAULT 0,
        cache_write_tokens  INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE state (
        key    TEXT PRIMARY KEY,
        value  TEXT NOT NULL
    );
    """,
)


MOOD_KEY = "mood"
SAMPLE_MIN_PEOPLE = 2
SAMPLE_MAX_CHARS = 40


class ExchangeMissing(LookupError):
    """The exchange a guess refers to does not exist (any more)."""


def normalize_phrase(text: str) -> str:
    """Lower-case, punctuation removed, whitespace collapsed: "Hungry?!" -> "hungry"."""
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() or ch == "'" else " " for ch in (text or "").lower())
    return " ".join(cleaned.split())


def common_phrases(guesses: list[tuple[str, str, float]], limit: int, min_people: int = SAMPLE_MIN_PEOPLE) -> tuple[str, ...]:
    """Normalised phrases guessed by at least ``min_people`` distinct people, most people first, clipped.

    ``guesses`` is ``(user_hash, text, score)``.
    """
    people: dict[str, set[str]] = {}
    scores: dict[str, list[float]] = {}
    for user_hash, text, score in guesses:
        phrase = normalize_phrase(text)
        if not phrase:
            continue
        people.setdefault(phrase, set()).add(user_hash)
        scores.setdefault(phrase, []).append(float(score))
    ranked = sorted(
        (p for p, who in people.items() if len(who) >= min_people),
        key=lambda p: (-len(people[p]), -sum(scores[p]) / len(scores[p]), p),
    )
    clip = lambda p: p if len(p) <= SAMPLE_MAX_CHARS else p[: SAMPLE_MAX_CHARS - 1] + "…"  # noqa: E731
    return tuple(clip(p) for p in ranked[:limit])


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class UserRow:
    user_hash: str
    keeping: bool
    onboarded: bool
    guesses_made: int


@dataclass(frozen=True)
class ExchangeRow:
    id: str
    user_hash: str
    source: str
    message_text: str
    intent: Intent
    glyph: Glyph
    model_version: str
    brain_source: str
    asleep: bool
    mood: str | None
    notice_shown: bool
    message_ref: str | None
    created_at: str


@dataclass(frozen=True)
class GuessRow:
    exchange_id: str
    user_hash: str
    guess_text: str
    score: float
    via: str
    created_at: str


@dataclass(frozen=True)
class DictionaryRow:
    intent: Intent
    guesses: int
    mean_score: float
    glyph: Glyph
    samples: tuple[str, ...]


@dataclass(frozen=True)
class SpendDay:
    day: str
    usd: float
    calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


@dataclass(frozen=True)
class ForgetResult:
    exchanges: int
    guesses_on_their_exchanges: int
    their_guesses: int
    user_row: bool


class Store:
    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            if str(path) != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._migrate()

    # -- plumbing -----------------------------------------------------------
    def _migrate(self) -> None:
        current = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if current > len(MIGRATIONS):
            raise RuntimeError(f"{self.path} has schema v{current}, newer than this code (v{len(MIGRATIONS)})")
        for version in range(current + 1, len(MIGRATIONS) + 1):
            self._conn.executescript("BEGIN;\n" + MIGRATIONS[version - 1] + f"\nPRAGMA user_version = {version};\nCOMMIT;")

    @property
    def schema_version(self) -> int:
        with self._lock:
            return self._conn.execute("PRAGMA user_version").fetchone()[0]

    def _tx(self):
        store = self

        class _Tx:
            def __enter__(self_inner):
                store._lock.acquire()
                store._conn.execute("BEGIN IMMEDIATE")
                return store._conn

            def __exit__(self_inner, exc_type, exc, tb):
                try:
                    store._conn.execute("ROLLBACK" if exc_type else "COMMIT")
                finally:
                    store._lock.release()
                return False

        return _Tx()

    def _one(self, sql: str, args: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, args).fetchone()

    def _all(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- users --------------------------------------------------------------
    def get_user(self, user_hash: str) -> UserRow | None:
        row = self._one("SELECT * FROM users WHERE user_hash = ?", (user_hash,))
        if row is None:
            return None
        return UserRow(row["user_hash"], bool(row["keeping"]), row["onboarded_at"] is not None, row["guesses_made"])

    def ensure_user(self, user_hash: str) -> UserRow:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO users (user_hash, created_at) VALUES (?, ?)", (user_hash, utcnow_iso())
            )
            return self.get_user(user_hash)  # type: ignore[return-value]

    def is_keeping(self, user_hash: str) -> bool:
        """Default is True: a user with no row yet is kept (disclosed by the first-time notice)."""
        user = self.get_user(user_hash)
        return True if user is None else user.keeping

    def set_keeping(self, user_hash: str, keeping: bool) -> None:
        with self._lock:
            self.ensure_user(user_hash)
            self._conn.execute("UPDATE users SET keeping = ? WHERE user_hash = ?", (int(keeping), user_hash))

    def mark_onboarded(self, user_hash: str) -> bool:
        """Record that the first-time notice was shown. Returns True only the first time."""
        with self._lock:
            self.ensure_user(user_hash)
            cur = self._conn.execute(
                "UPDATE users SET onboarded_at = ? WHERE user_hash = ? AND onboarded_at IS NULL",
                (utcnow_iso(), user_hash),
            )
            return cur.rowcount == 1

    def bump_guesses_made(self, user_hash: str) -> int:
        with self._lock:
            self.ensure_user(user_hash)
            self._conn.execute("UPDATE users SET guesses_made = guesses_made + 1 WHERE user_hash = ?", (user_hash,))
            return self._one("SELECT guesses_made FROM users WHERE user_hash = ?", (user_hash,))[0]

    # -- exchanges ----------------------------------------------------------
    def add_exchange(
        self,
        *,
        exchange_id: str,
        user_hash: str,
        source: str,
        message_text: str,
        intent: Intent,
        glyph: Glyph,
        brain_source: str,
        asleep: bool,
        mood: str | None,
        notice_shown: bool,
    ) -> bool:
        """Store an exchange, unless its author is not being kept. Returns whether it was stored.

        The keeping check and the insert happen in one transaction, so a "Stop
        keeping" (from the bot or the operator CLI, in any process) that lands
        while the brain was thinking can never be overridden by this write.
        """
        with self._tx() as conn:
            row = conn.execute("SELECT keeping FROM users WHERE user_hash = ?", (user_hash,)).fetchone()
            if row is not None and not row["keeping"]:
                return False
            if row is None:
                conn.execute("INSERT INTO users (user_hash, created_at) VALUES (?, ?)", (user_hash, utcnow_iso()))
            conn.execute(
                "INSERT INTO exchanges (id, user_hash, source, message_text, intent, glyph, model_version, "
                "brain_source, asleep, mood, notice_shown, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    exchange_id, user_hash, source, message_text, intent.to_json(), glyph.to_json(),
                    glyph.model_version, brain_source, int(asleep), mood, int(notice_shown), utcnow_iso(),
                ),
            )
            return True

    def set_message_ref(self, exchange_id: str, message_ref: str) -> bool:
        with self._lock:
            cur = self._conn.execute("UPDATE exchanges SET message_ref = ? WHERE id = ?", (message_ref, exchange_id))
            return cur.rowcount == 1

    @staticmethod
    def _exchange(row: sqlite3.Row) -> ExchangeRow:
        return ExchangeRow(
            id=row["id"], user_hash=row["user_hash"], source=row["source"], message_text=row["message_text"],
            intent=Intent.from_json(row["intent"]), glyph=Glyph.from_json(row["glyph"]),
            model_version=row["model_version"], brain_source=row["brain_source"], asleep=bool(row["asleep"]),
            mood=row["mood"], notice_shown=bool(row["notice_shown"]), message_ref=row["message_ref"],
            created_at=row["created_at"],
        )

    def get_exchange(self, exchange_id: str) -> ExchangeRow | None:
        row = self._one("SELECT * FROM exchanges WHERE id = ?", (exchange_id,))
        return None if row is None else self._exchange(row)

    def exchange_by_message_ref(self, message_ref: str) -> ExchangeRow | None:
        row = self._one("SELECT * FROM exchanges WHERE message_ref = ?", (message_ref,))
        return None if row is None else self._exchange(row)

    # -- guesses ------------------------------------------------------------
    def add_guess(self, *, exchange_id: str, user_hash: str, guess_text: str, score: float, via: str) -> bool:
        """Store a guess, in one transaction with its checks.

        Returns False (and stores nothing) if this user already guessed this
        exchange or is not being kept. Raises ``ExchangeMissing`` if the
        exchange no longer exists (its owner deleted it meanwhile).
        """
        try:
            with self._tx() as conn:
                if conn.execute("SELECT 1 FROM exchanges WHERE id = ?", (exchange_id,)).fetchone() is None:
                    raise ExchangeMissing(exchange_id)
                row = conn.execute("SELECT keeping FROM users WHERE user_hash = ?", (user_hash,)).fetchone()
                if row is not None and not row["keeping"]:
                    return False
                if row is None:
                    conn.execute("INSERT INTO users (user_hash, created_at) VALUES (?, ?)", (user_hash, utcnow_iso()))
                cur = conn.execute(
                    "INSERT OR IGNORE INTO guesses (exchange_id, user_hash, guess_text, score, via, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (exchange_id, user_hash, guess_text, float(score), via, utcnow_iso()),
                )
                return cur.rowcount == 1
        except sqlite3.IntegrityError as exc:  # foreign key: the exchange vanished under us
            raise ExchangeMissing(exchange_id) from exc

    def get_guess(self, exchange_id: str, user_hash: str) -> GuessRow | None:
        row = self._one("SELECT * FROM guesses WHERE exchange_id = ? AND user_hash = ?", (exchange_id, user_hash))
        if row is None:
            return None
        return GuessRow(row["exchange_id"], row["user_hash"], row["guess_text"], row["score"], row["via"], row["created_at"])

    def guess_count(self, exchange_id: str) -> int:
        return self._one("SELECT COUNT(*) FROM guesses WHERE exchange_id = ?", (exchange_id,))[0]

    # -- per-user counts and forgetting ------------------------------------
    def user_counts(self, user_hash: str) -> tuple[int, int]:
        """(stored messages, stored guesses) for this person."""
        ex = self._one("SELECT COUNT(*) FROM exchanges WHERE user_hash = ?", (user_hash,))[0]
        gu = self._one("SELECT COUNT(*) FROM guesses WHERE user_hash = ?", (user_hash,))[0]
        return ex, gu

    def forget(self, user_hash: str) -> ForgetResult:
        """Delete everything tied to ``user_hash``.

        Removes their exchanges (with every guess anyone made on them, via the
        cascade), their own guesses on other people's glyphs, and their user
        row, and resets Skeebert's global mood to the default (it may have
        been shaped by their messages). The one exception is the keeping
        preference: if they had pressed "Stop keeping", a bare row
        ``(user_hash, keeping=0)`` is kept so that deleting does not silently
        switch keeping back on.
        """
        with self._tx() as conn:
            on_theirs = conn.execute(
                "SELECT COUNT(*) FROM guesses WHERE exchange_id IN (SELECT id FROM exchanges WHERE user_hash = ?)",
                (user_hash,),
            ).fetchone()[0]
            exchanges = conn.execute("DELETE FROM exchanges WHERE user_hash = ?", (user_hash,)).rowcount
            theirs = conn.execute("DELETE FROM guesses WHERE user_hash = ?", (user_hash,)).rowcount
            row = conn.execute("SELECT keeping FROM users WHERE user_hash = ?", (user_hash,)).fetchone()
            conn.execute("DELETE FROM users WHERE user_hash = ?", (user_hash,))
            # Skeebert's mood may have been nudged by this person's messages; reset it rather than guess.
            conn.execute("DELETE FROM state WHERE key = ?", (MOOD_KEY,))
            if row is not None and not row["keeping"]:
                conn.execute(
                    "INSERT INTO users (user_hash, keeping, created_at) VALUES (?, 0, ?)", (user_hash, utcnow_iso())
                )
        return ForgetResult(exchanges, on_theirs, theirs, row is not None)

    # -- training -----------------------------------------------------------
    def iter_training_examples(self) -> Iterator[TrainingExample]:
        rows = self._all(
            "SELECT e.intent, e.glyph, g.guess_text, g.score FROM guesses g "
            "JOIN exchanges e ON e.id = g.exchange_id ORDER BY g.id"
        )
        for row in rows:
            yield TrainingExample(
                intent=Intent.from_json(row["intent"]),
                glyph=Glyph.from_json(row["glyph"]),
                guess_text=row["guess_text"],
                score=min(1.0, max(0.0, float(row["score"]))),
            )

    def training_examples(self) -> list[TrainingExample]:
        """Every stored guess as a ``TrainingExample`` with the exact glyph that was shown."""
        return list(self.iter_training_examples())

    # -- dictionary ---------------------------------------------------------
    def dictionary(self, model_version: str, min_guesses: int = 3, limit: int = 9, samples: int = 2) -> list[DictionaryRow]:
        """Intents drawn by ``model_version`` with at least ``min_guesses`` guesses, best mean score first.

        ``samples`` are only phrases that at least two different people
        guessed (after normalising), never attributed, so one person's words
        are never quoted to other servers.
        """
        rows = self._all(
            "SELECT e.intent AS intent, COUNT(g.id) AS n, AVG(g.score) AS mean, MAX(e.created_at) AS last "
            "FROM exchanges e JOIN guesses g ON g.exchange_id = e.id "
            "WHERE e.model_version = ? GROUP BY e.intent HAVING COUNT(g.id) >= ? "
            "ORDER BY mean DESC, n DESC, e.intent ASC LIMIT ?",
            (model_version, min_guesses, limit),
        )
        out = []
        for row in rows:
            glyph_row = self._one(
                "SELECT glyph FROM exchanges WHERE model_version = ? AND intent = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (model_version, row["intent"]),
            )
            sample_rows = self._all(
                "SELECT g.user_hash, g.guess_text, g.score FROM guesses g JOIN exchanges e ON e.id = g.exchange_id "
                "WHERE e.model_version = ? AND e.intent = ? AND TRIM(g.guess_text) != ''",
                (model_version, row["intent"]),
            )
            picked = common_phrases([(r["user_hash"], r["guess_text"], r["score"]) for r in sample_rows], samples)
            out.append(
                DictionaryRow(
                    intent=Intent.from_json(row["intent"]), guesses=row["n"], mean_score=float(row["mean"]),
                    glyph=Glyph.from_json(glyph_row["glyph"]), samples=tuple(picked),
                )
            )
        return out

    # -- spend --------------------------------------------------------------
    def add_spend(
        self, day: str, usd: float, *, input_tokens: int, output_tokens: int, cache_read_tokens: int, cache_write_tokens: int
    ) -> float:
        """Add one call's cost to ``day``'s ledger. Returns the day's new total."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO spend (day, usd, calls, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens) "
                "VALUES (?, ?, 1, ?, ?, ?, ?) ON CONFLICT(day) DO UPDATE SET usd = usd + excluded.usd, "
                "calls = calls + 1, input_tokens = input_tokens + excluded.input_tokens, "
                "output_tokens = output_tokens + excluded.output_tokens, "
                "cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens, "
                "cache_write_tokens = cache_write_tokens + excluded.cache_write_tokens",
                (day, float(usd), input_tokens, output_tokens, cache_read_tokens, cache_write_tokens),
            )
            return self.spent_on(day)

    def spent_on(self, day: str) -> float:
        row = self._one("SELECT usd FROM spend WHERE day = ?", (day,))
        return 0.0 if row is None else float(row["usd"])

    def spend_days(self, limit: int = 14) -> list[SpendDay]:
        rows = self._all("SELECT * FROM spend ORDER BY day DESC LIMIT ?", (limit,))
        return [
            SpendDay(r["day"], float(r["usd"]), r["calls"], r["input_tokens"], r["output_tokens"],
                     r["cache_read_tokens"], r["cache_write_tokens"])
            for r in rows
        ]

    # -- state --------------------------------------------------------------
    def get_state(self, key: str, default: str | None = None) -> str | None:
        row = self._one("SELECT value FROM state WHERE key = ?", (key,))
        return default if row is None else row["value"]

    def set_state(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def is_empty(self) -> bool:
        """True when nothing hashed has ever been stored (safe to create a new salt)."""
        return all(
            self._one(f"SELECT COUNT(*) FROM {t}")[0] == 0 for t in ("users", "exchanges", "guesses")
        )

    # -- summary ------------------------------------------------------------
    def summary(self) -> dict:
        def count(sql: str) -> int:
            return self._one(sql)[0]

        return {
            "users": count("SELECT COUNT(*) FROM users"),
            "users_not_keeping": count("SELECT COUNT(*) FROM users WHERE keeping = 0"),
            "exchanges": count("SELECT COUNT(*) FROM exchanges"),
            "exchanges_asleep": count("SELECT COUNT(*) FROM exchanges WHERE asleep = 1"),
            "guesses": count("SELECT COUNT(*) FROM guesses"),
            "guesses_with_text": count("SELECT COUNT(*) FROM guesses WHERE TRIM(guess_text) != ''"),
            "distinct_intents": count("SELECT COUNT(DISTINCT intent) FROM exchanges"),
            "model_versions": [
                (r["model_version"], r["n"])
                for r in self._all("SELECT model_version, COUNT(*) AS n FROM exchanges GROUP BY model_version ORDER BY n DESC")
            ],
            "mean_score": (lambda v: None if v is None else float(v))(self._one("SELECT AVG(score) FROM guesses")[0]),
        }

