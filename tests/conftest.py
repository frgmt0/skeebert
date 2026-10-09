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
