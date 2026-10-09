import json

import pytest
from conftest import FakeAnthropic, api_connection_error, api_status_error, make_message

from skeebert import concepts
from skeebert.brain import BrainRequest, HaikuBrain, LocalBrain, Turn, build_system_prompt, feeling_atoms
from skeebert.brain.haiku import BadBrainOutput, build_schema, parse_output
from skeebert.budget import Budget, RateLimiter
from skeebert.config import Prices
from skeebert.store import Store
from skeebert.types import Intent


@pytest.fixture
def parts(settings, embedder):
    store = Store(settings.db_path)
    budget = Budget(store, settings.daily_budget_usd, settings.prices)
    limiter = RateLimiter(settings.rate_limit_per_hour)
    local = LocalBrain(embedder)
    return store, budget, limiter, local


def haiku(client, settings, parts, **kw):
    store, budget, limiter, local = parts
    return HaikuBrain(client, settings, kw.get("budget", budget), kw.get("limiter", limiter), local)


# -- local brain -------------------------------------------------------------

def test_local_brain_picks_nearest_atoms(embedder):
    brain = LocalBrain(embedder)
    t = brain.think(BrainRequest("I want some food"))
    assert "food" in t.intent.concepts
    assert "tired" in t.intent.concepts  # sleepy marker
    assert t.source == "local" and t.asleep and t.mood == "tired"
    assert 1 <= len(t.intent.concepts) <= 3


def test_local_brain_handles_empty_message(embedder):
    t = LocalBrain(embedder).think(BrainRequest(""))
    assert t.intent == Intent.of("greeting", "tired")


def test_local_brain_awake_mode_keeps_mood(embedder):
    t = LocalBrain(embedder).think(BrainRequest("music", mood="happy"), asleep=False)
    assert "tired" not in t.intent.concepts and t.mood == "happy" and not t.asleep


def test_local_brain_respects_vocabulary(embedder):
    brain = LocalBrain(embedder, vocabulary=["food", "water", "happy"])
    t = brain.think(BrainRequest("some water please"))
    assert set(t.intent.concepts) <= {"food", "water", "happy"}
    assert brain.sleepy_atom is None


# -- prompt & schema ---------------------------------------------------------

def test_system_prompt_is_stable_and_lists_every_atom():
    a, b = build_system_prompt(), build_system_prompt()
    assert a == b
    for c in concepts.CONCEPTS:
        assert f"{c.name}: {c.gloss}" in a


def test_schema_enums():
    schema = build_schema(concepts.names())
    assert schema["properties"]["atoms"]["items"]["enum"] == concepts.names()
    assert schema["properties"]["mood"]["enum"] == feeling_atoms()
    assert schema["additionalProperties"] is False


def test_parse_output_validation():
    names = concepts.names()
    assert parse_output('{"atoms": ["question", "food"], "mood": "happy"}', names) == (Intent.of("food", "question"), "happy")
    assert parse_output('{"atoms": ["food", "food", "yes", "no", "eat"], "mood": "happy"}', names)[0] == Intent.of("food", "yes", "no")
    for bad in ['{"atoms": ["blorp"], "mood": "happy"}', '{"atoms": [], "mood": "happy"}',
                '{"atoms": ["food"], "mood": "food"}', "not json", '["food"]', None]:
        with pytest.raises(BadBrainOutput):
            parse_output(bad, names)


# -- haiku brain -------------------------------------------------------------

def test_haiku_parses_structured_output_and_records_spend(settings, parts):
    store, budget, *_ = parts
    client = FakeAnthropic(make_message({"atoms": ["food", "question"], "mood": "excited"},
                                        input_tokens=150, output_tokens=60, cache_read=3200, cache_write=0))
    brain = haiku(client, settings, parts)
    req = BrainRequest("are you hungry?", mood="curious", history=(Turn("hi", Intent.of("greeting")),), user_key="u_1")
    t = brain.think(req)
    assert t.intent == Intent.of("food", "question") and t.mood == "excited"
    assert t.source == "haiku" and not t.asleep
    expected = (150 * 0.10 + 60 * 0.50 + 3200 * 0.01) / 1e6
    assert budget.spent_today() == pytest.approx(expected)
    [call] = client.messages.calls
    assert call["model"] == "claude-haiku-5-5"
    assert "thinking" not in call
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["system"][0]["text"] == build_system_prompt()
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["effort"] == "low"
    user = call["messages"][0]["content"]
    assert "are you hungry?" in user and "Skeebert meant: greeting" in user and "current mood: curious" in user


def test_haiku_unknown_atom_falls_back_to_local_but_records_spend(settings, parts):
    store, budget, *_ = parts
    brain = haiku(FakeAnthropic(make_message({"atoms": ["zorp"], "mood": "happy"})), settings, parts)
    t = brain.think(BrainRequest("food please", user_key="u"))
    assert t.source == "local" and t.asleep and t.reason == "bad_output"
    assert budget.spent_today() > 0


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_haiku_bad_stop_reason_falls_back(settings, parts, stop):
    brain = haiku(FakeAnthropic(make_message({"atoms": ["food"], "mood": "happy"}, stop_reason=stop)), settings, parts)
    t = brain.think(BrainRequest("food", user_key="u"))
    assert t.source == "local" and t.reason == "bad_output"


@pytest.mark.parametrize("error", [api_status_error(529), api_status_error(500), api_connection_error()])
def test_haiku_api_error_falls_back_and_spends_nothing(settings, parts, error):
    store, budget, *_ = parts
    brain = haiku(FakeAnthropic(error), settings, parts)
    t = brain.think(BrainRequest("hello there", user_key="u"))
    assert t.source == "local" and t.asleep and t.reason == "api_error"
    assert budget.spent_today() == 0.0
    assert budget.remaining_today() == pytest.approx(settings.daily_budget_usd)  # reservation released


def test_haiku_budget_cap_sleeps_without_calling(settings, parts):
    store, _, limiter, local = parts
    budget = Budget(store, 0.0, Prices())
    client = FakeAnthropic(make_message({"atoms": ["food"], "mood": "happy"}))
    t = HaikuBrain(client, settings, budget, limiter, local).think(BrainRequest("food", user_key="u"))
    assert t.reason == "budget" and t.asleep and client.messages.calls == []
    assert limiter.try_acquire("u")  # slot was refunded


def test_haiku_rate_limit_per_user(settings, parts):
    store, budget, _, local = parts
    limiter = RateLimiter(2)
    client = FakeAnthropic(make_message({"atoms": ["happy"], "mood": "happy"}))
    brain = HaikuBrain(client, settings, budget, limiter, local)
    assert brain.think(BrainRequest("a", user_key="u1")).source == "haiku"
    assert brain.think(BrainRequest("b", user_key="u1")).source == "haiku"
    third = brain.think(BrainRequest("c", user_key="u1"))
    assert third.source == "local" and third.reason == "rate_limited" and third.asleep
    assert brain.think(BrainRequest("d", user_key="u2")).source == "haiku"
    assert len(client.messages.calls) == 3


def test_haiku_request_stays_far_under_100k(settings, parts):
    client = FakeAnthropic(make_message({"atoms": ["happy"], "mood": "happy"}))
    brain = haiku(client, settings, parts)
    huge = "word " * 50_000
    history = tuple(Turn("blah " * 5000, Intent.of("happy")) for _ in range(20))
    brain.think(BrainRequest(huge, history=history, user_key="u"))
    call = client.messages.calls[0]
    text = call["system"][0]["text"] + call["messages"][0]["content"] + json.dumps(call["output_config"])
    assert len(text) // 3 < settings.brain_max_input_tokens < 100_000
    assert call["messages"][0]["content"].count("someone said") <= settings.context_messages


def test_haiku_vocabulary_restricts_schema(settings, parts):
    store, budget, limiter, local = parts
    vocab = concepts.names()[:50]
    brain = HaikuBrain(FakeAnthropic(make_message({"atoms": ["food"], "mood": "happy"})), settings, budget, limiter, local, vocab)
    assert "food" not in vocab
    assert brain.think(BrainRequest("x", user_key="u")).reason == "bad_output"


def test_timeout_charges_worst_case_reservation(settings, parts):
    import anthropic
    import httpx2

    store, budget, *_ = parts
    err = anthropic.APITimeoutError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    brain = haiku(FakeAnthropic(err), settings, parts)
    t = brain.think(BrainRequest("hello", user_key="u"))
    assert t.reason == "api_error" and t.asleep
    _, prompt_tokens = brain.build_user_turn(BrainRequest("hello", user_key="u"))
    assert budget.spent_today() == pytest.approx(budget.max_cost(prompt_tokens, settings.brain_max_tokens))


def test_real_client_has_no_hidden_retries(settings, parts):
    store, budget, limiter, local = parts
    brain = HaikuBrain.from_settings(settings.with_overrides(anthropic_api_key="sk-test"), budget, limiter, local)
    assert brain.client.max_retries == 0


def test_estimate_tokens_is_a_byte_upper_bound():
    from skeebert.brain.haiku import estimate_tokens

    assert estimate_tokens("abc") >= 3
    emoji = "🍔" * 100  # 4 bytes each; a tokenizer can spend a token per byte
    assert estimate_tokens(emoji) >= 400
