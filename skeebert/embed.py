"""Sentence embeddings for comparing what a human guessed with what Skeebert meant.

``Embedder`` wraps sentence-transformers' ``all-MiniLM-L6-v2`` (384-d, small
enough for a laptop CPU). Anything with a matching ``dim`` and ``encode``
satisfies ``TextEmbedder``, which is how the tests inject a deterministic,
offline stand-in instead of downloading a model.
"""

from __future__ import annotations

import threading
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIM = 384


@runtime_checkable
class TextEmbedder(Protocol):
    """Maps texts to L2-normalised float32 vectors of shape [len(texts), dim]."""

    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class Embedder:
    """Lazy wrapper around a sentence-transformers model.

    The model is loaded on first ``encode`` rather than at construction so that
    importing the bot or building an engine never triggers a download or a
    multi-second load unless embeddings are actually needed.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None, dim: int = DEFAULT_DIM):
        self.model_name = model_name
        self.device = device
        self.dim = dim
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(self.model_name, device=self.device)
                # renamed in sentence-transformers 6; keep working on older releases too
                getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
                actual = getter()
                if actual != self.dim:
                    raise ValueError(f"{self.model_name} produces {actual}-d vectors, expected {self.dim}")
                self._model = model
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._load()
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return l2_normalize(np.asarray(vecs, dtype=np.float32))


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, eps)
