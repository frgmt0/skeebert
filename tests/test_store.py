import threading

import pytest

from skeebert.store import MIGRATIONS, Store
from skeebert.types import Intent


def _add(store, engine, xid, user, atoms=("food", "question")):
    intent = Intent(atoms)
    store.add_exchange(
        exchange_id=xid, user_hash=user, source="mention", message_text=f"msg {xid}", intent=intent,
        glyph=engine.speak(intent), brain_source="haiku", asleep=False, mood="curious", notice_shown=False,
    )


def test_schema_and_migrations(tmp_path):
    s = Store(tmp_path / "x.db")
    assert s.schema_version == len(MIGRATIONS)
    s.close()
    s2 = Store(tmp_path / "x.db")  # reopening does not re-run migrations
    assert s2.schema_version == len(MIGRATIONS)
    tables = {r[0] for r in s2._all("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"users", "exchanges", "guesses", "spend", "state"} <= tables


def test_keeping_default_and_toggle(tmp_path):
    s = Store(tmp_path / "x.db")
    assert s.is_keeping("u_a") is True
    s.set_keeping("u_a", False)
    assert s.is_keeping("u_a") is False
    s.set_keeping("u_a", True)
    assert s.is_keeping("u_a") is True


def test_onboarding_only_once(tmp_path):
    s = Store(tmp_path / "x.db")
    assert s.mark_onboarded("u_a") is True
    assert s.mark_onboarded("u_a") is False
    assert s.get_user("u_a").onboarded


def test_exchange_roundtrip_keeps_exact_glyph(tmp_path, engine):
    s = Store(tmp_path / "x.db")
    _add(s, engine, "aaaaaaaaaaaa", "u_a")
    row = s.get_exchange("aaaaaaaaaaaa")
    assert row.intent == Intent.of("food", "question")
    assert row.glyph == engine.speak(Intent.of("food", "question"))
    assert s.set_message_ref("aaaaaaaaaaaa", "m_1")
    assert s.exchange_by_message_ref("m_1").id == "aaaaaaaaaaaa"
    assert s.exchange_by_message_ref("m_2") is None


def test_one_guess_per_user_per_exchange(tmp_path, engine):
    s = Store(tmp_path / "x.db")
    _add(s, engine, "aaaaaaaaaaaa", "u_a")
    assert s.add_guess(exchange_id="aaaaaaaaaaaa", user_hash="u_b", guess_text="food?", score=0.7, via="button")
    assert not s.add_guess(exchange_id="aaaaaaaaaaaa", user_hash="u_b", guess_text="again", score=0.1, via="reply")
    assert s.get_guess("aaaaaaaaaaaa", "u_b").guess_text == "food?"
    assert s.guess_count("aaaaaaaaaaaa") == 1


def test_forget_cascades(tmp_path, engine):
    s = Store(tmp_path / "x.db")
    _add(s, engine, "aaaaaaaaaaaa", "u_a")
    _add(s, engine, "bbbbbbbbbbbb", "u_b", ("happy",))
    s.add_guess(exchange_id="aaaaaaaaaaaa", user_hash="u_b", guess_text="b on a", score=0.5, via="button")
    s.add_guess(exchange_id="aaaaaaaaaaaa", user_hash="u_c", guess_text="c on a", score=0.5, via="button")
    s.add_guess(exchange_id="bbbbbbbbbbbb", user_hash="u_a", guess_text="a on b", score=0.5, via="button")
    s.add_guess(exchange_id="bbbbbbbbbbbb", user_hash="u_c", guess_text="c on b", score=0.5, via="button")
    r = s.forget("u_a")
    assert (r.exchanges, r.guesses_on_their_exchanges, r.their_guesses) == (1, 2, 1)
    assert s.get_exchange("aaaaaaaaaaaa") is None
    assert s.user_counts("u_a") == (0, 0)
    assert s.get_user("u_a") is None
    remaining = [e.guess_text for e in s.training_examples()]
    assert remaining == ["c on b"]
    assert s.get_exchange("bbbbbbbbbbbb") is not None


def test_forget_preserves_stop_keeping(tmp_path, engine):
    s = Store(tmp_path / "x.db")
    _add(s, engine, "aaaaaaaaaaaa", "u_a")
    s.set_keeping("u_a", False)
    s.forget("u_a")
    assert s.is_keeping("u_a") is False
    assert s.user_counts("u_a") == (0, 0)


def test_training_examples_use_stored_glyph(tmp_path, engine):
    s = Store(tmp_path / "x.db")
    _add(s, engine, "aaaaaaaaaaaa", "u_a")
    s.add_guess(exchange_id="aaaaaaaaaaaa", user_hash="u_b", guess_text="hungry?", score=0.8, via="reply")
    [ex] = s.training_examples()
    assert ex.intent == Intent.of("food", "question")
    assert ex.glyph == s.get_exchange("aaaaaaaaaaaa").glyph
    assert ex.guess_text == "hungry?" and ex.score == pytest.approx(0.8)


def test_dictionary_threshold_and_ranking(tmp_path, engine):
    s = Store(tmp_path / "x.db")
    v = engine.model_version
    _add(s, engine, "aaaaaaaaaaaa", "u_a", ("food",))
    _add(s, engine, "bbbbbbbbbbbb", "u_a", ("happy",))
    _add(s, engine, "cccccccccccc", "u_a", ("sad",))
    for i, sc in enumerate([0.9, 0.8, 0.7]):
        s.add_guess(exchange_id="aaaaaaaaaaaa", user_hash=f"u_{i}", guess_text=f"food {i}", score=sc, via="button")
    for i, sc in enumerate([0.95, 0.95, 0.95, 0.95]):
        s.add_guess(exchange_id="bbbbbbbbbbbb", user_hash=f"u_{i}", guess_text="joy" if i else "JOY!", score=sc, via="button")
    for i in range(2):  # below threshold
        s.add_guess(exchange_id="cccccccccccc", user_hash=f"u_{i}", guess_text="sad", score=1.0, via="button")
    rows = s.dictionary(v)
    assert [str(r.intent) for r in rows] == ["happy", "food"]
    assert rows[0].guesses == 4 and rows[0].mean_score == pytest.approx(0.95)
    assert rows[0].samples == ("joy",)  # normalised: "JOY!" and "joy" are one phrase, 4 people
    assert rows[1].samples == ()  # every food phrase came from a single person: never quoted
    assert s.dictionary("someotherversion") == []


def test_spend_and_state(tmp_path):
    s = Store(tmp_path / "x.db")
    assert s.spent_on("2026-10-08") == 0.0
    s.add_spend("2026-10-08", 0.5, input_tokens=1, output_tokens=2, cache_read_tokens=3, cache_write_tokens=4)
    total = s.add_spend("2026-10-08", 0.25, input_tokens=1, output_tokens=2, cache_read_tokens=3, cache_write_tokens=4)
    assert total == pytest.approx(0.75)
    [day] = s.spend_days()
    assert day.calls == 2 and day.cache_write_tokens == 8
    s.set_state("mood", "happy")
    s.set_state("mood", "sad")
    assert s.get_state("mood") == "sad"
    assert s.get_state("nope", "x") == "x"


def test_store_is_thread_safe(tmp_path, engine):
    s = Store(tmp_path / "x.db")
    _add(s, engine, "aaaaaaaaaaaa", "u_a")
    errors = []

    def worker(i):
        try:
            for j in range(20):
                s.add_guess(exchange_id="aaaaaaaaaaaa", user_hash=f"u_{i}_{j}", guess_text="x", score=0.1, via="reply")
                s.add_spend("d", 0.01, input_tokens=1, output_tokens=1, cache_read_tokens=0, cache_write_tokens=0)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert s.guess_count("aaaaaaaaaaaa") == 160
    assert s.spent_on("d") == pytest.approx(1.6)


def test_common_phrases_need_two_people_and_are_clipped():
    from skeebert.store import common_phrases, normalize_phrase

    assert normalize_phrase("  Are you HUNGRY?! ") == "are you hungry"
    long = "a very long guess that goes on and on and on forever"
    guesses = [("u1", "Hungry?", 0.9), ("u2", "hungry", 0.8), ("u1", "hungry!!", 0.9),  # 2 people
               ("u3", "secret phrase", 1.0), ("u3", "Secret phrase", 1.0),  # same person twice: not enough
               ("u4", long, 0.5), ("u5", long, 0.5), ("u6", "", 0.0), ("u7", "  ", 0.0)]
    out = common_phrases(guesses, limit=5)
    assert out[0] == "hungry"
    assert "secret phrase" not in out
    assert out[1].endswith("…") and len(out[1]) == 40


def test_add_exchange_rechecks_keeping_atomically(tmp_path, engine):
    s = Store(tmp_path / "x.db")
    s.set_keeping("u_a", False)
    intent = Intent.of("food")
    stored = s.add_exchange(exchange_id="aaaaaaaaaaaa", user_hash="u_a", source="mention", message_text="m",
                            intent=intent, glyph=engine.speak(intent), brain_source="local", asleep=True,
                            mood="tired", notice_shown=False)
    assert stored is False and s.user_counts("u_a") == (0, 0)


def test_add_guess_on_missing_exchange_raises_exchange_missing(tmp_path, engine):
    from skeebert.store import ExchangeMissing

    s = Store(tmp_path / "x.db")
    _add(s, engine, "aaaaaaaaaaaa", "u_a")
    s.forget("u_a")
    with pytest.raises(ExchangeMissing):
        s.add_guess(exchange_id="aaaaaaaaaaaa", user_hash="u_b", guess_text="x", score=0.1, via="button")
    s.set_keeping("u_c", False)
    _add(s, engine, "bbbbbbbbbbbb", "u_d")
    assert s.add_guess(exchange_id="bbbbbbbbbbbb", user_hash="u_c", guess_text="x", score=0.1, via="button") is False
    assert s.guess_count("bbbbbbbbbbbb") == 0


def test_forget_resets_mood(tmp_path, engine):
    s = Store(tmp_path / "x.db")
    s.set_state("mood", "angry")
    s.forget("u_nobody")
    assert s.get_state("mood") is None


def test_is_empty(tmp_path):
    s = Store(tmp_path / "x.db")
    assert s.is_empty()
    s.set_keeping("u_a", False)
    assert not s.is_empty()
