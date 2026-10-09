"""Everything Skeebert does, with no Discord in it.

The Discord adapter (``bot.py``) and the terminal (``cli.py play``) are thin
shells around ``Skeebert``. They pass in raw ids and text and get back plain
dataclasses (bytes for images, strings for message text), which is what makes
the whole behaviour testable without a token.

A turn
------
1. Someone addresses Skeebert. ``handle_talk`` hashes their id, gathers up to
   six recent Skeebert-addressed messages from the same conversation (an
   in-memory ring buffer, never persisted) and the current mood, and asks the
   brain for a ``Thought``: 1-3 atoms plus a mood.
2. The glyph engine draws the intent. The reply is the PNG, a Guess button,
   and almost no text: 💤 if the brain was asleep, and, the very first time a
   person talks to Skeebert, one line linking the privacy page.
3. Unless the person pressed "Stop keeping", the exchange (message text,
   intent, exact glyph, model version, brain source) is stored and the
   message joins the context buffer. If they did (even while the brain was
   still thinking), it is held only in an in-memory LRU so other people's
   guesses can still be scored and revealed; nothing derived from their
   message is written or kept as context.
4. ``handle_guess`` scores a guess (``scoring.score_guess``) and returns the
   closeness tier, the reveal (``Intent.gloss()``) and how many people have
   guessed. One guess per person per exchange counts; asking again returns
   the first result. A guess is stored only if both the exchange is stored
   and the guesser is keeping. Someone whose first-ever interaction is a
   guess gets the one-line privacy notice with it (a reply guess gets it as
   a short reply of its own).

What the site promises (https://skeebert.frgmt.xyz/privacy) is binding on
this module: kept by default, only things addressed to Skeebert, salted
hashes, self-serve stop and delete.
"""

from __future__ import annotations

import logging
import secrets
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from . import concepts
from .brain import Brain, BrainRequest, DEFAULT_MOOD, Thought, Turn
from .budget import Budget, RateLimiter
from .config import Settings
from .embed import TextEmbedder
from .identity import Pseudonymiser
from .scoring import score_guess
from .sheet import MAX_GLYPHS, contact_sheet_png
from .store import MOOD_KEY, DictionaryRow, ExchangeMissing, ForgetResult, Store
from .types import Glyph, Intent

log = logging.getLogger(__name__)

ASLEEP_MARKER = "💤"
DICTIONARY_MIN_GUESSES = 3
TIP_FIRST = "tip: you can also just reply to a glyph with your guess"
TIP_THIRD = "tip: /skeebert dictionary shows what's been cracked"
EPHEMERAL_LIMIT = 2000  # exchanges / guesses held in memory for people who stopped keeping
CONVERSATIONS_LIMIT = 1000  # ring buffers held in memory

# (lowest score, word, emoji), best first
TIERS: tuple[tuple[float, str, str], ...] = (
    (0.75, "nailed it", "🎯"),
    (0.50, "close", "🔥"),
    (0.25, "warm", "🤏"),
    (0.00, "cold", "🧊"),
)


def tier_for(score: float) -> tuple[str, str]:
    for floor, word, emoji in TIERS:
        if score >= floor:
            return word, emoji
    return TIERS[-1][1], TIERS[-1][2]


def score_bar(score: float, width: int = 10) -> str:
    filled = max(0, min(width, round(score * width)))
    return "▰" * filled + "▱" * (width - filled)


def first_time_line(privacy_url: str) -> str:
    return f"first time? skeebert talks in pictures. tap **Guess** to decode it · what it keeps: <{privacy_url}>"


def intent_label(intent: Intent) -> str:
    return " + ".join(intent.concepts)


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ExchangeGone(LookupError):
    """The exchange was deleted by its owner, or was held only in memory and has expired."""


@dataclass(frozen=True)
class Exchange:
    id: str
    user_hash: str
    intent: Intent
    glyph: Glyph
    asleep: bool
    notice_shown: bool
    kept: bool


@dataclass(frozen=True)
class TalkReply:
    exchange_id: str
    png: bytes
    content: str
    thought: Thought
    glyph: Glyph
    kept: bool
    first_time: bool


@dataclass(frozen=True)
class GuessOutcome:
    exchange_id: str
    guess_text: str
    score: float
    tier_word: str
    tier_emoji: str
    intent: Intent
    guessers: int
    already: bool = False
    tip: str | None = None
    notice: str | None = None

    @property
    def meaning(self) -> str:
        return self.intent.gloss()

    def render(self) -> str:
        lines = [
            f"{self.tier_emoji} **{self.tier_word}** `{score_bar(self.score)}` {self.score:.2f}",
            f"you said: “{_clip(self.guess_text, 120)}”" if self.guess_text.strip() else "you said nothing",
            f"skeebert meant: **{intent_label(self.intent)}** ({self.meaning})",
            f"decoded by {self.guessers} so far",
        ]
        if self.already:
            lines.insert(0, "you already guessed this one; only your first guess counts.")
        if self.tip:
            lines.append(self.tip)
        if self.notice:
            lines.append(self.notice)
        return "\n".join(lines)


@dataclass(frozen=True)
class PrivacyStatus:
    keeping: bool
    messages: int
    guesses: int

    def render(self, privacy_url: str) -> str:
        if self.keeping:
            head = "skeebert is **keeping** what you send it (messages to it, your guesses, the glyphs it made you)."
        else:
            head = "skeebert is **not keeping** anything new from you. it still talks to you."
        return (
            f"{head}\nstored right now: {self.messages} message{'s' if self.messages != 1 else ''}, "
            f"{self.guesses} guess{'es' if self.guesses != 1 else ''}.\nfull policy: <{privacy_url}>"
        )


@dataclass(frozen=True)
class DictionaryEntry:
    number: int
    row: DictionaryRow

    def render(self) -> str:
        r = self.row
        line = (
            f"{self.number} · {intent_label(r.intent)} — avg {r.mean_score:.2f} over {r.guesses} "
            f"guess{'es' if r.guesses != 1 else ''}"
        )
        if r.samples:
            line += " — people said: " + ", ".join(f"“{_clip(s, 40)}”" for s in r.samples)
        return line


@dataclass(frozen=True)
class Dictionary:
    model_version: str
    entries: tuple[DictionaryEntry, ...]
    png: bytes | None

    def render(self) -> str:
        if not self.entries:
            return (
                "nothing has been cracked yet. a glyph shows up here once it has at least "
                f"{DICTIONARY_MIN_GUESSES} guesses on the current model."
            )
        return "\n".join(e.render() for e in self.entries)


def build_dictionary(store: Store, model_version: str, min_guesses: int = DICTIONARY_MIN_GUESSES) -> Dictionary:
    rows = store.dictionary(model_version, min_guesses=min_guesses, limit=MAX_GLYPHS)
    entries = tuple(DictionaryEntry(i + 1, r) for i, r in enumerate(rows))
    png = contact_sheet_png([r.glyph for r in rows]) if rows else None
    return Dictionary(model_version, entries, png)


def help_text(site_url: str) -> str:
    return "\n".join(
        [
            "**skeebert** talks in glyphs. you work out what they mean.",
            "talk: @skeebert <message>, a DM, or `/skeebert say`",
            "guess: tap **Guess** under a glyph, or reply to the glyph",
            "`/skeebert dictionary`: what's been cracked · `/skeebert privacy`: what's kept, stop or delete",
            f"more: <{site_url}>",
        ]
    )


@dataclass
class _HistoryEntry:
    user_hash: str
    text: str
    reply: Intent


@dataclass
class _MemoryGuess:
    guess_text: str
    score: float
    via: str


@dataclass
class _Memory:
    """Everything held only in RAM: never written anywhere, gone on restart."""

    exchanges: "OrderedDict[str, Exchange]" = field(default_factory=OrderedDict)
    refs: dict[str, str] = field(default_factory=dict)  # message_ref -> exchange id (ephemeral exchanges)
    guesses: "OrderedDict[tuple[str, str], _MemoryGuess]" = field(default_factory=OrderedDict)
    guess_counts: dict[str, int] = field(default_factory=dict)  # exchange id -> unpersisted guesses
    guesses_made: dict[str, int] = field(default_factory=dict)  # user_hash -> guesses (non-keeping users)
    history: "OrderedDict[str, deque[_HistoryEntry]]" = field(default_factory=OrderedDict)


class Skeebert:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        ident: Pseudonymiser,
        engine,
        embedder: TextEmbedder,
        brain: Brain,
        *,
        budget: Budget | None = None,
        limiter: RateLimiter | None = None,
        ephemeral_limit: int = EPHEMERAL_LIMIT,
    ) -> None:
        self.settings = settings
        self.store = store
        self.ident = ident
        self.engine = engine
        self.embedder = embedder
        self.brain = brain
        self.budget = budget
        self.limiter = limiter
        self.ephemeral_limit = ephemeral_limit
        self._mem = _Memory()
        self._lock = threading.RLock()
        self._engine_lock = threading.Lock()
        self._epochs: dict[str, int] = {}  # user_hash -> bumped on stop-keeping and delete

    # -- helpers -------------------------------------------------------------
    @property
    def model_version(self) -> str:
        return self.engine.model_version

    def user_hash(self, user_id: object) -> str:
        return self.ident.user(user_id)

    def mood(self) -> str:
        mood = self.store.get_state(MOOD_KEY, DEFAULT_MOOD) or DEFAULT_MOOD
        return mood if concepts.is_known(mood) else DEFAULT_MOOD

    def speak(self, intent: Intent) -> Glyph:
        with self._engine_lock:
            return self.engine.speak(intent)

    def _remember_ephemeral(self, ex: Exchange) -> None:
        with self._lock:
            mem = self._mem
            mem.exchanges[ex.id] = ex
            mem.exchanges.move_to_end(ex.id)
            while len(mem.exchanges) > self.ephemeral_limit:
                old_id, _ = mem.exchanges.popitem(last=False)
                self._drop_ephemeral_exchange(old_id)

    def _drop_ephemeral_exchange(self, xid: str) -> None:
        mem = self._mem
        mem.exchanges.pop(xid, None)
        for ref in [r for r, x in mem.refs.items() if x == xid]:
            del mem.refs[ref]
        for key in [k for k in mem.guesses if k[0] == xid]:
            del mem.guesses[key]
        mem.guess_counts.pop(xid, None)

    def _history(self, conversation_key: str) -> deque:
        mem = self._mem
        buf = mem.history.get(conversation_key)
        if buf is None:
            buf = deque(maxlen=max(1, self.settings.context_messages))
            mem.history[conversation_key] = buf
            while len(mem.history) > CONVERSATIONS_LIMIT:
                mem.history.popitem(last=False)
        mem.history.move_to_end(conversation_key)
        return buf

    # -- talking -------------------------------------------------------------
    def handle_talk(self, user_id: object, text: str, *, source: str, conversation_key: str) -> TalkReply:
        """Answer something addressed to Skeebert. Blocking (brain call, embedding, render): run off the event loop.

        The brain call can take seconds. If the person presses "Stop keeping"
        or "Delete everything" meanwhile, that wins: their per-user epoch has
        moved, so nothing from this message is written to disk or added to
        the context buffer (the reply is still sent and its glyph is held in
        memory only). The store re-checks keeping inside the insert's own
        transaction as well, which also covers the operator CLI.
        """
        uh = self.user_hash(user_id)
        text = (text or "").strip()
        with self._lock:
            epoch = self._epochs.get(uh, 0)
            history = tuple(Turn(e.text, e.reply) for e in self._history(conversation_key))
        thought = self.brain.think(BrainRequest(message=text, mood=self.mood(), history=history, user_key=uh))
        glyph = self.speak(thought.intent)
        xid = secrets.token_hex(6)
        with self._lock:  # serialised against set_keeping / forget_hash in this process
            unchanged = self._epochs.get(uh, 0) == epoch
            if unchanged:
                first_time = self.store.mark_onboarded(uh)
            else:  # they just stopped keeping or deleted: show the notice if due, but write nothing
                user = self.store.get_user(uh)
                first_time = user is None or not user.onboarded
            kept = False
            if unchanged:
                kept = self.store.add_exchange(
                    exchange_id=xid, user_hash=uh, source=source, message_text=text, intent=thought.intent,
                    glyph=glyph, brain_source=thought.source, asleep=thought.asleep, mood=thought.mood,
                    notice_shown=first_time,
                )
            if kept:
                if thought.source == "haiku":
                    self.store.set_state(MOOD_KEY, thought.mood)
                self._history(conversation_key).append(_HistoryEntry(uh, text, thought.intent))
            else:
                self._remember_ephemeral(Exchange(xid, uh, thought.intent, glyph, thought.asleep, first_time, kept=False))
        ex = Exchange(xid, uh, thought.intent, glyph, thought.asleep, first_time, kept=kept)
        return TalkReply(
            exchange_id=xid, png=glyph.png_bytes(), content=self._content(ex), thought=thought, glyph=glyph,
            kept=kept, first_time=first_time,
        )

    def _content(self, ex: Exchange) -> str:
        head = []
        if ex.asleep:
            head.append(ASLEEP_MARKER)
        n = self.guess_count(ex.id)
        if n:
            head.append(f"decoded by {n}")
        lines = [" · ".join(head)] if head else []
        if ex.notice_shown:
            lines.append(first_time_line(self.settings.privacy_url))
        return "\n".join(lines)

    def message_content(self, exchange_id: str) -> str:
        """The glyph message's text right now (marker, guess counter, first-time line)."""
        ex = self.get_exchange(exchange_id)
        if ex is None:
            raise ExchangeGone(exchange_id)
        return self._content(ex)

    def attach_message(self, exchange_id: str, message_id: object) -> None:
        """Remember which Discord message shows this exchange's glyph, so replies to it count as guesses."""
        ref = self.ident.message(message_id)
        if self.store.set_message_ref(exchange_id, ref):
            return
        with self._lock:
            if exchange_id in self._mem.exchanges:
                self._mem.refs[ref] = exchange_id

    def get_exchange(self, exchange_id: str) -> Exchange | None:
        row = self.store.get_exchange(exchange_id)
        if row is not None:
            return Exchange(row.id, row.user_hash, row.intent, row.glyph, row.asleep, row.notice_shown, kept=True)
        with self._lock:
            return self._mem.exchanges.get(exchange_id)

    def exchange_for_message(self, message_id: object) -> Exchange | None:
        ref = self.ident.message(message_id)
        row = self.store.exchange_by_message_ref(ref)
        if row is not None:
            return Exchange(row.id, row.user_hash, row.intent, row.glyph, row.asleep, row.notice_shown, kept=True)
        with self._lock:
            xid = self._mem.refs.get(ref)
            return self._mem.exchanges.get(xid) if xid else None

    # -- guessing ------------------------------------------------------------
    def guess_count(self, exchange_id: str) -> int:
        with self._lock:
            extra = self._mem.guess_counts.get(exchange_id, 0)
        return self.store.guess_count(exchange_id) + extra

    def existing_guess(self, user_id: object, exchange_id: str) -> GuessOutcome | None:
        """This person's earlier (counted) guess on this exchange, as a result to show again."""
        ex = self.get_exchange(exchange_id)
        if ex is None:
            raise ExchangeGone(exchange_id)
        return self._existing(self.user_hash(user_id), ex)

    def _existing(self, uh: str, ex: Exchange) -> GuessOutcome | None:
        row = self.store.get_guess(ex.id, uh)
        if row is not None:
            text, score = row.guess_text, row.score
        else:
            with self._lock:
                mg = self._mem.guesses.get((ex.id, uh))
            if mg is None:
                return None
            text, score = mg.guess_text, mg.score
        word, emoji = tier_for(score)
        return GuessOutcome(ex.id, text, score, word, emoji, ex.intent, self.guess_count(ex.id), already=True)

    def handle_guess(self, user_id: object, exchange_id: str, text: str, *, via: str) -> GuessOutcome:
        """Score a guess. Returns the earlier result (``already=True``) if this person already guessed.

        ``via`` is ``button``, ``reply`` or ``cli``. Blocking (embedding): run off the event loop.
        Raises ``ExchangeGone`` if the exchange no longer exists, including when
        its owner deleted it while this guess was being scored. Like talking,
        a stop/delete by the guesser during scoring means nothing is written.
        """
        ex = self.get_exchange(exchange_id)
        if ex is None:
            raise ExchangeGone(exchange_id)
        uh = self.user_hash(user_id)
        prior = self._existing(uh, ex)
        if prior is not None:
            return prior
        with self._lock:
            epoch = self._epochs.get(uh, 0)
        text = (text or "").strip()
        score = score_guess(ex.intent, text, self.embedder)
        with self._lock:
            if self.get_exchange(ex.id) is None:
                raise ExchangeGone(exchange_id)
            unchanged = self._epochs.get(uh, 0) == epoch
            stored = False
            if ex.kept and unchanged:
                try:
                    stored = self.store.add_guess(exchange_id=ex.id, user_hash=uh, guess_text=text, score=score, via=via)
                except ExchangeMissing:
                    raise ExchangeGone(exchange_id) from None
                if not stored:
                    prior = self._existing(uh, ex)  # their own concurrent guess got there first
                    if prior is not None:
                        return prior
            if not stored:  # not kept (exchange or guesser): memory only
                key = (ex.id, uh)
                if key in self._mem.guesses:
                    return self._existing(uh, ex)  # type: ignore[return-value]
                self._mem.guesses[key] = _MemoryGuess(text, score, via)
                self._mem.guess_counts[ex.id] = self._mem.guess_counts.get(ex.id, 0) + 1
                while len(self._mem.guesses) > self.ephemeral_limit:
                    (old_x, _), _ = self._mem.guesses.popitem(last=False)
                    self._decrement_count(old_x)
            guesser_kept = unchanged and self.store.is_keeping(uh)
            made = self._bump_guesses_made(uh, guesser_kept)
            notice = None
            if via in ("button", "reply") and unchanged and self.store.mark_onboarded(uh):
                notice = first_time_line(self.settings.privacy_url)
        tip = None
        if via == "button":
            if made == 1:
                tip = TIP_FIRST
            elif made == 3:
                tip = TIP_THIRD
        word, emoji = tier_for(score)
        return GuessOutcome(ex.id, text, score, word, emoji, ex.intent, self.guess_count(ex.id), tip=tip, notice=notice)

    def _decrement_count(self, xid: str) -> None:
        left = self._mem.guess_counts.get(xid, 0) - 1
        if left > 0:
            self._mem.guess_counts[xid] = left
        else:
            self._mem.guess_counts.pop(xid, None)

    def _bump_guesses_made(self, uh: str, keeping: bool) -> int:
        if keeping:
            return self.store.bump_guesses_made(uh)
        with self._lock:
            n = self._mem.guesses_made.get(uh, 0) + 1
            self._mem.guesses_made[uh] = n
            return n

    # -- privacy -------------------------------------------------------------
    def privacy_status(self, user_id: object) -> PrivacyStatus:
        uh = self.user_hash(user_id)
        messages, guesses = self.store.user_counts(uh)
        return PrivacyStatus(self.store.is_keeping(uh), messages, guesses)

    def set_keeping(self, user_id: object, keeping: bool) -> PrivacyStatus:
        """Stop or restart keeping. Stopping does not delete what is already stored (that's ``delete_everything``)."""
        uh = self.user_hash(user_id)
        with self._lock:
            if not keeping:
                self._epochs[uh] = self._epochs.get(uh, 0) + 1  # in-flight talks/guesses must not write
                for buf in self._mem.history.values():  # and their recent words leave the context buffer
                    kept = [e for e in buf if e.user_hash != uh]
                    buf.clear()
                    buf.extend(kept)
            self.store.set_keeping(uh, keeping)
        return self.privacy_status(user_id)

    def delete_everything(self, user_id: object) -> ForgetResult:
        return self.forget_hash(self.user_hash(user_id))

    def forget_hash(self, uh: str) -> ForgetResult:
        """Delete everything tied to this pseudonym, on disk and in memory. The keeping preference survives.

        Skeebert's global mood is reset to the default (the store does it),
        since it may have been shaped by this person's messages. The hourly
        rate limit is deliberately *not* reset: deleting must not be a way
        around it.
        """
        with self._lock:
            self._epochs[uh] = self._epochs.get(uh, 0) + 1
            result = self.store.forget(uh)
            mem = self._mem
            for xid in [x for x, ex in mem.exchanges.items() if ex.user_hash == uh]:
                self._drop_ephemeral_exchange(xid)
            for key in [k for k in mem.guesses if k[1] == uh]:
                del mem.guesses[key]
                self._decrement_count(key[0])
            mem.guesses_made.pop(uh, None)
            for buf in mem.history.values():
                kept = [e for e in buf if e.user_hash != uh]
                buf.clear()
                buf.extend(kept)
        return result

    # -- dictionary & help ---------------------------------------------------
    def dictionary(self) -> Dictionary:
        return build_dictionary(self.store, self.model_version)

    def help(self) -> str:
        return help_text(self.settings.site_url)


def build_service(
    settings: Settings,
    *,
    embedder: TextEmbedder | None = None,
    anthropic_client=None,
    engine=None,
) -> Skeebert:
    """Wire the real pieces together from settings. Loads (or creates) the served checkpoint."""
    from .brain import HaikuBrain, LocalBrain
    from .glyph import GlyphEngine

    store = Store(settings.db_path)
    ident = Pseudonymiser.load(settings.salt_path, create=store.is_empty())
    if engine is None:
        engine = GlyphEngine.load(settings.checkpoint_dir, device=settings.device)
    vocabulary = concepts.names()[: engine.n_concepts]
    if embedder is None:
        from .embed import Embedder

        embedder = Embedder(settings.embed_model)
    local = LocalBrain(embedder, vocabulary)
    limiter = RateLimiter(settings.rate_limit_per_hour)
    budget = Budget(store, settings.daily_budget_usd, settings.prices)
    if anthropic_client is not None:
        brain: Brain = HaikuBrain(anthropic_client, settings, budget, limiter, local, vocabulary)
    elif settings.anthropic_api_key:
        brain = HaikuBrain.from_settings(settings, budget, limiter, local, vocabulary)
    else:
        log.info("no ANTHROPIC_API_KEY: skeebert runs on the local (sleeping) brain only")
        brain = local
    return Skeebert(settings, store, ident, engine, embedder, brain, budget=budget, limiter=limiter)
