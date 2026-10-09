import pytest
from conftest import FakeAnthropic, make_message

from skeebert.brain import LocalBrain
from skeebert.service import (
    ASLEEP_MARKER, TIP_FIRST, TIP_THIRD, ExchangeGone, Skeebert, build_service, first_time_line, score_bar, tier_for,
)
from skeebert.types import Intent


class FixedBrain:
    """Always means the same thing, awake, like a Haiku brain that answered."""

    def __init__(self, atoms=("food", "question"), mood="happy", asleep=False):
        self.atoms, self.mood, self.asleep = atoms, mood, asleep
        self.requests = []

    def think(self, request):
        from skeebert.brain import Thought

        self.requests.append(request)
        return Thought(Intent(self.atoms), self.mood, "local" if self.asleep else "haiku", self.asleep)


def _parts(settings, engine, embedder):
    from skeebert.identity import Pseudonymiser
    from skeebert.store import Store

    return Store(settings.db_path), Pseudonymiser.load(settings.salt_path), engine, embedder


def make(settings, engine, embedder, brain=None, **kw) -> Skeebert:
    return Skeebert(settings, *_parts(settings, engine, embedder), brain or FixedBrain(), **kw)


def talk(s, user=1, text="are you hungry?", key="channel:1"):
    return s.handle_talk(user, text, source="mention", conversation_key=key)


def test_talk_returns_png_and_stores_exchange(settings, engine, embedder):
    s = make(settings, engine, embedder)
    r = talk(s)
    assert r.png.startswith(b"\x89PNG")
    assert r.kept and r.thought.intent == Intent.of("food", "question")
    row = s.store.get_exchange(r.exchange_id)
    assert row.message_text == "are you hungry?"
    assert row.glyph == engine.speak(Intent.of("food", "question"))
    assert row.user_hash == s.user_hash(1) and "1" != row.user_hash
    assert s.mood() == "happy"  # haiku mood persisted


def test_first_time_notice_exactly_once(settings, engine, embedder):
    s = make(settings, engine, embedder)
    first = talk(s, user=7)
    assert first.first_time
    assert first.content == first_time_line(settings.privacy_url)
    assert "https://skeebert.frgmt.xyz/privacy" in first.content
    second = talk(s, user=7)
    assert not second.first_time and second.content == ""
    # survives a restart (new service, same data dir)
    s2 = make(settings, engine, embedder)
    assert talk(s2, user=7).content == ""
    assert talk(s2, user=8).first_time


def test_asleep_marker(settings, engine, embedder):
    s = make(settings, engine, embedder, brain=LocalBrain(embedder))
    talk(s, user=1)
    r = talk(s, user=1, text="food")
    assert r.content == ASLEEP_MARKER
    assert "tired" in r.thought.intent.concepts
    assert s.mood() == "curious"  # sleeping brain never overwrites the persistent mood


def test_context_ring_buffer_is_per_conversation_and_bounded(settings, engine, embedder):
    brain = FixedBrain()
    s = make(settings, engine, embedder, brain=brain)
    for i in range(10):
        talk(s, user=i, text=f"msg {i}", key="channel:a")
    talk(s, user=99, text="other", key="channel:b")
    talk(s, user=1, text="last", key="channel:a")
    hist = brain.requests[-1].history
    assert len(hist) == settings.context_messages
    assert [t.text for t in hist] == [f"msg {i}" for i in range(4, 10)]
    assert all(t.reply == Intent.of("food", "question") for t in hist)
    assert brain.requests[-1].mood == "happy"
    assert brain.requests[-1].user_key == s.user_hash(1)


def test_guess_scores_reveals_and_counts(settings, engine, embedder):
    s = make(settings, engine, embedder)
    r = talk(s, user=1)
    talk(s, user=2)  # onboard user 2 so no notice in the outcome
    o = s.handle_guess(2, r.exchange_id, "food question", via="button")
    assert 0.0 <= o.score <= 1.0 and o.score > 0.5
    assert o.meaning == Intent.of("food", "question").gloss()
    assert o.guessers == 1 and not o.already
    text = o.render()
    assert "skeebert meant: **food + question**" in text and "decoded by 1" in text
    assert s.message_content(r.exchange_id).splitlines() == ["decoded by 1", first_time_line(settings.privacy_url)]
    assert s.store.get_guess(r.exchange_id, s.user_hash(2)).guess_text == "food question"


def test_one_guess_per_exchange(settings, engine, embedder):
    s = make(settings, engine, embedder)
    r = talk(s, user=1)
    first = s.handle_guess(2, r.exchange_id, "food", via="reply")
    again = s.handle_guess(2, r.exchange_id, "food question exactly", via="button")
    assert again.already and again.score == first.score and again.guess_text == "food"
    assert s.existing_guess(2, r.exchange_id).already
    assert s.existing_guess(3, r.exchange_id) is None
    assert s.guess_count(r.exchange_id) == 1


def test_reply_guess_tier(settings, engine, embedder):
    s = make(settings, engine, embedder)
    r = talk(s, user=1)
    talk(s, user=2)  # 2 has met skeebert; 3 has not
    good = s.handle_guess(2, r.exchange_id, "food asking a question", via="reply")
    bad = s.handle_guess(3, r.exchange_id, "zebra trombone", via="reply")
    assert good.tier_emoji in ("🎯", "🔥") and bad.tier_emoji == "🧊"
    assert good.tip is None and good.notice is None  # replies get a reaction only
    assert bad.notice == first_time_line(settings.privacy_url)  # ...plus the notice, the first time ever
    assert s.store.get_user(s.user_hash(3)).onboarded
    assert s.guess_count(r.exchange_id) == 2


def test_tiers_and_bar():
    assert tier_for(0.9) == ("nailed it", "🎯")
    assert tier_for(0.5) == ("close", "🔥")
    assert tier_for(0.3) == ("warm", "🤏")
    assert tier_for(0.0) == ("cold", "🧊")
    assert score_bar(0.0) == "▱" * 10 and score_bar(1.0) == "▰" * 10 and score_bar(0.44) == "▰" * 4 + "▱" * 6


def test_tips_only_on_first_and_third_guess(settings, engine, embedder):
    s = make(settings, engine, embedder)
    talk(s, user=5)
    tips = []
    for i in range(5):
        r = talk(s, user=1)
        tips.append(s.handle_guess(5, r.exchange_id, "food", via="button").tip)
    assert tips == [TIP_FIRST, None, TIP_THIRD, None, None]


def test_guess_button_shows_notice_to_never_onboarded_user(settings, engine, embedder):
    s = make(settings, engine, embedder)
    r = talk(s, user=1)
    o = s.handle_guess(42, r.exchange_id, "food", via="button")
    assert o.notice == first_time_line(settings.privacy_url)
    r2 = talk(s, user=1)
    assert s.handle_guess(42, r2.exchange_id, "food", via="button").notice is None
    assert not talk(s, user=42).first_time


def test_stop_keeping_persists_nothing_from_them(settings, engine, embedder):
    s = make(settings, engine, embedder)
    talk(s, user=1)  # onboard
    s.set_keeping(1, False)
    before = s.store.summary()
    r = talk(s, user=1, text="secret stuff")
    assert not r.kept
    assert s.store.get_exchange(r.exchange_id) is None
    # others can still guess and get the reveal, but those guesses aren't persisted either
    o = s.handle_guess(2, r.exchange_id, "food", via="button")
    assert o.meaning == Intent.of("food", "question").gloss() and o.guessers == 1
    assert s.handle_guess(2, r.exchange_id, "other", via="button").already
    assert s.message_content(r.exchange_id) == "decoded by 1"
    # their own guesses on kept glyphs aren't persisted
    kept = talk(s, user=3)
    s.handle_guess(1, kept.exchange_id, "food", via="button")
    assert s.store.get_guess(kept.exchange_id, s.user_hash(1)) is None
    assert s.guess_count(kept.exchange_id) == 1
    after = s.store.summary()
    assert after["exchanges"] == before["exchanges"] + 1  # only user 3's exchange
    assert after["guesses"] == before["guesses"]
    assert "secret stuff" not in open(settings.db_path, "rb").read().decode("latin-1")
    assert s.mood() == "happy"  # set by user 1's first (kept) talk, and by user 3, not by the unkept one


def test_reply_mapping_for_kept_and_unkept(settings, engine, embedder):
    s = make(settings, engine, embedder)
    r = talk(s, user=1)
    s.attach_message(r.exchange_id, 111)
    assert s.exchange_for_message(111).id == r.exchange_id
    assert s.store.exchange_by_message_ref(s.ident.message(111)) is not None
    s.set_keeping(9, False)
    r2 = talk(s, user=9)
    s.attach_message(r2.exchange_id, 222)
    assert s.exchange_for_message(222).id == r2.exchange_id
    assert s.store.exchange_by_message_ref(s.ident.message(222)) is None
    assert s.exchange_for_message(333) is None


def test_privacy_toggle_and_delete(settings, engine, embedder):
    s = make(settings, engine, embedder)
    r = talk(s, user=1)
    other = talk(s, user=2)
    s.handle_guess(2, r.exchange_id, "food", via="button")
    s.handle_guess(1, other.exchange_id, "food", via="button")
    st = s.privacy_status(1)
    assert st.keeping and (st.messages, st.guesses) == (1, 1)
    assert "keeping" in st.render(settings.privacy_url)
    st = s.set_keeping(1, False)
    assert not st.keeping and st.messages == 1  # stopping doesn't delete
    assert "not keeping" in st.render(settings.privacy_url)
    res = s.delete_everything(1)
    assert (res.exchanges, res.guesses_on_their_exchanges, res.their_guesses) == (1, 1, 1)
    st = s.privacy_status(1)
    assert (st.keeping, st.messages, st.guesses) == (False, 0, 0)  # keeping preference survives delete
    with pytest.raises(ExchangeGone):
        s.handle_guess(3, r.exchange_id, "x", via="button")
    assert s.guess_count(other.exchange_id) == 0
    assert s.set_keeping(1, True).keeping


def test_delete_purges_memory_too(settings, engine, embedder):
    brain = FixedBrain()
    s = make(settings, engine, embedder, brain=brain)
    s.set_keeping(1, False)
    r = talk(s, user=1, text="private words", key="c")
    s.attach_message(r.exchange_id, 5)
    s.delete_everything(1)
    assert s.get_exchange(r.exchange_id) is None and s.exchange_for_message(5) is None
    talk(s, user=2, text="hi", key="c")
    assert all(t.text != "private words" for t in brain.requests[-1].history)


def test_ephemeral_lru_is_bounded(settings, engine, embedder):
    s = make(settings, engine, embedder, ephemeral_limit=3)
    s.set_keeping(1, False)
    ids = [talk(s, user=1).exchange_id for _ in range(5)]
    assert s.get_exchange(ids[0]) is None and s.get_exchange(ids[-1]) is not None


def test_dictionary_threshold_and_render(settings, engine, embedder):
    s = make(settings, engine, embedder)
    d = s.dictionary()
    assert d.entries == () and d.png is None and "nothing has been cracked yet" in d.render()
    r = talk(s, user=1)
    for u in (2, 3):
        s.handle_guess(u, r.exchange_id, "food question", via="reply")
    assert s.dictionary().entries == ()
    s.handle_guess(4, r.exchange_id, "hungry?", via="reply")
    d = s.dictionary()
    assert len(d.entries) == 1 and d.png.startswith(b"\x89PNG")
    line = d.render()
    assert line.startswith("1 · food + question — avg ") and "over 3 guesses" in line
    assert "people said: “food question”" in line  # said by 2 people
    assert "hungry" not in line  # said by only one person: never quoted


def test_help_is_short(settings, engine, embedder):
    text = make(settings, engine, embedder).help()
    assert len(text.splitlines()) <= 5 and "https://skeebert.frgmt.xyz" in text


def test_build_service_uses_haiku_when_client_given(settings, engine, embedder):
    client = FakeAnthropic(make_message({"atoms": ["greeting"], "mood": "happy"}))
    s = build_service(settings, embedder=embedder, engine=engine, anthropic_client=client)
    r = talk(s, text="hello!")
    assert r.thought.source == "haiku" and r.thought.intent == Intent.of("greeting")
    assert s.budget.spent_today() > 0
    local = build_service(settings, embedder=embedder, engine=engine)
    assert isinstance(local.brain, LocalBrain)


# -- regressions from the review ----------------------------------------------

class HookBrain(FixedBrain):
    """Runs ``hook`` while "thinking", i.e. during the slow brain call."""

    def __init__(self, hook, **kw):
        super().__init__(**kw)
        self.hook = hook

    def think(self, request):
        self.hook()
        return super().think(request)


def test_stop_keeping_during_brain_call_wins(settings, engine, embedder):
    holder = {}
    brain = HookBrain(lambda: holder["s"].set_keeping(1, False))
    s = holder["s"] = make(settings, engine, embedder, brain=brain)
    r = talk(s, user=1, text="private", key="c")
    assert not r.kept
    assert s.store.user_counts(s.user_hash(1)) == (0, 0)
    assert s.get_exchange(r.exchange_id) is not None  # still guessable, from memory
    brain.hook = lambda: None
    talk(s, user=2, text="next", key="c")
    assert all(t.text != "private" for t in brain.requests[-1].history)


def test_delete_during_brain_call_is_not_undone(settings, engine, embedder):
    holder = {}
    brain = HookBrain(lambda: holder["s"].delete_everything(1))
    s = holder["s"] = make(settings, engine, embedder, brain=brain)
    talk(s, user=1, text="first", key="c")
    assert s.store.user_counts(s.user_hash(1)) == (0, 0)
    assert s.store.get_user(s.user_hash(1)) is None  # nothing re-created, not even an onboarded row
    brain.hook = lambda: None
    talk(s, user=2, key="c")
    assert all(t.text != "first" for t in brain.requests[-1].history)


def test_stop_keeping_during_guess_scoring_wins(settings, engine, embedder, monkeypatch):
    import skeebert.service as service

    s = make(settings, engine, embedder)
    r = talk(s, user=1)
    real = service.score_guess

    def slow_score(*a, **k):
        s.set_keeping(2, False)
        return real(*a, **k)

    monkeypatch.setattr(service, "score_guess", slow_score)
    o = s.handle_guess(2, r.exchange_id, "food", via="button")
    assert o.guessers == 1
    assert s.store.get_guess(r.exchange_id, s.user_hash(2)) is None


def test_owner_delete_during_guess_scoring_is_exchange_gone(settings, engine, embedder, monkeypatch):
    import skeebert.service as service

    s = make(settings, engine, embedder)
    r = talk(s, user=1)
    real = service.score_guess

    def slow_score(*a, **k):
        s.delete_everything(1)
        return real(*a, **k)

    monkeypatch.setattr(service, "score_guess", slow_score)
    with pytest.raises(ExchangeGone):
        s.handle_guess(2, r.exchange_id, "food", via="button")


def test_unkept_owner_delete_during_scoring_is_exchange_gone(settings, engine, embedder, monkeypatch):
    import skeebert.service as service

    s = make(settings, engine, embedder)
    s.set_keeping(1, False)
    r = talk(s, user=1)
    real = service.score_guess
    monkeypatch.setattr(service, "score_guess", lambda *a, **k: (s.delete_everything(1), real(*a, **k))[1])
    with pytest.raises(ExchangeGone):
        s.handle_guess(2, r.exchange_id, "food", via="button")


def test_not_kept_messages_never_join_context(settings, engine, embedder):
    brain = FixedBrain()
    s = make(settings, engine, embedder, brain=brain)
    s.set_keeping(1, False)
    talk(s, user=1, text="unkept words", key="c")
    talk(s, user=2, text="kept words", key="c")
    talk(s, user=2, text="again", key="c")
    assert [t.text for t in brain.requests[-1].history] == ["kept words"]


def test_stop_keeping_clears_their_context(settings, engine, embedder):
    brain = FixedBrain()
    s = make(settings, engine, embedder, brain=brain)
    talk(s, user=1, text="said before stopping", key="c")
    s.set_keeping(1, False)
    talk(s, user=2, text="x", key="c")
    assert all(t.text != "said before stopping" for t in brain.requests[-1].history)


def test_delete_does_not_reset_rate_limit(settings, engine, embedder):
    from skeebert.budget import RateLimiter

    lim = RateLimiter(2)
    s = make(settings, engine, embedder, limiter=lim)
    uh = s.user_hash(1)
    assert lim.try_acquire(uh) and lim.try_acquire(uh) and not lim.try_acquire(uh)
    s.delete_everything(1)
    assert not lim.try_acquire(uh)


def test_delete_resets_mood(settings, engine, embedder):
    s = make(settings, engine, embedder, brain=FixedBrain(mood="angry"))
    talk(s, user=1)
    assert s.mood() == "angry"
    s.delete_everything(2)  # anyone's delete resets it
    assert s.mood() == "curious"


def test_build_service_refuses_new_salt_for_existing_data(settings, engine, embedder):
    from skeebert.identity import SaltMissing

    s = build_service(settings, embedder=embedder, engine=engine)
    talk(s, user=1)
    settings.salt_path.unlink()
    with pytest.raises(SaltMissing):
        build_service(settings, embedder=embedder, engine=engine)
    assert not settings.salt_path.exists()
