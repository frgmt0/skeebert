"""Shared test fixtures. The fake embedder lives here, never in the package."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from skeebert.model import ModelConfig


class HashingEmbedder:
    """Deterministic, offline bag-of-words embedder with the real model's 384-d shape.

    Each lower-cased word maps to a fixed pseudo-random unit vector (seeded by
    its sha256); a text is the normalised sum of its words. Texts sharing words
    are therefore similar and texts sharing none are near-orthogonal, which is
    enough structure to test scoring order and to give training real targets.
    """

    dim = 384

    def __init__(self):
        self.calls = 0

    def _word(self, word: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for w in re.findall(r"[a-z]+", text.lower()):
                out[i] += self._word(w)
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def tiny_config():
    return ModelConfig.tiny()


# ---------------------------------------------------------------------------
# application-layer fixtures (no network, no Discord, never the real data/ dir)
# ---------------------------------------------------------------------------

import json  # noqa: E402

from skeebert.config import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    return Settings(
        data_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "checkpoints",
        renders_dir=tmp_path / "renders",
    )


@pytest.fixture
def engine(settings, tiny_config):
    from skeebert.glyph import GlyphEngine

    return GlyphEngine.load(settings.checkpoint_dir, config=tiny_config)


def make_message(payload, *, input_tokens=120, output_tokens=40, cache_read=3000, cache_write=0,
                 stop_reason="end_turn", model="claude-haiku-5-5"):
    """A real ``anthropic.types.Message``, shaped like a structured-output response."""
    from anthropic.types import Message, TextBlock, Usage

    text = payload if isinstance(payload, str) else json.dumps(payload)
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model=model,
        content=[TextBlock(type="text", text=text)],
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ),
    )


def api_status_error(status=529, message="Overloaded"):
    import anthropic
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    return anthropic.InternalServerError(message, response=response, body=None)


def api_connection_error():
    import anthropic
    import httpx2

    return anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(kwargs)
        return item


class FakeAnthropic:
    """Stands in for ``anthropic.Anthropic``: ``.messages.create(**kwargs)`` returns queued responses."""

    def __init__(self, *responses):
        self.messages = FakeMessages(responses)
