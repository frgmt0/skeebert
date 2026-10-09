"""The sleeping brain: pick the atoms whose glosses sit closest to the message.

No network, no cost. The message is embedded with the same sentence model
used for scoring guesses, compared with every atom's gloss, and the nearest
one (or two, when the runner-up is nearly as close) becomes the intent. When
Skeebert is asleep the ``tired`` atom is added, so a sleepy answer is visibly
a sleepy answer to anyone who learns that part of the language.

This is a reflex, not understanding: it echoes the topic of the message back
rather than replying to it. That is the honest trade for not spending money.
"""

from __future__ import annotations

import threading
from typing import Sequence

import numpy as np

from .. import concepts
from ..embed import TextEmbedder, l2_normalize
from ..types import MAX_INTENT_ATOMS, Intent
from .base import DEFAULT_MOOD, BrainRequest, Thought

SLEEPY_ATOM = "tired"
EMPTY_MESSAGE_ATOM = "greeting"
SECOND_ATOM_RATIO = 0.85  # runner-up joins if it is at least this fraction as close as the best
SECOND_ATOM_MIN = 0.20  # ...and at least this close in absolute cosine


class LocalBrain:
    def __init__(self, embedder: TextEmbedder, vocabulary: Sequence[str] | None = None) -> None:
        self.embedder = embedder
        names = list(vocabulary) if vocabulary is not None else concepts.names()
        for n in names:
            concepts.by_name(n)  # raises on unknown atoms
        self.vocabulary = names
        self.sleepy_atom = SLEEPY_ATOM if SLEEPY_ATOM in names else None
        self._candidates = [n for n in names if n != self.sleepy_atom]
        self._atom_vecs: np.ndarray | None = None
        self._lock = threading.Lock()

    def _vectors(self) -> np.ndarray:
        with self._lock:
            if self._atom_vecs is None:
                glosses = [concepts.by_name(n).gloss for n in self._candidates]
                self._atom_vecs = l2_normalize(self.embedder.encode(glosses))
            return self._atom_vecs

    def nearest(self, text: str, k: int = 2) -> list[tuple[str, float]]:
        """The ``k`` candidate atoms closest to ``text`` with their cosines, best first."""
        vec = l2_normalize(self.embedder.encode([text]))[0]
        sims = self._vectors() @ vec
        order = np.argsort(-sims, kind="stable")[:k]
        return [(self._candidates[i], float(sims[i])) for i in order]

    def pick_atoms(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return [EMPTY_MESSAGE_ATOM] if EMPTY_MESSAGE_ATOM in self._candidates else [self._candidates[0]]
        ranked = self.nearest(text, k=2)
        atoms = [ranked[0][0]]
        if len(ranked) > 1:
            (_, best), (second, s2) = ranked[0], ranked[1]
            if s2 >= SECOND_ATOM_MIN and best > 0 and s2 >= SECOND_ATOM_RATIO * best:
                atoms.append(second)
        return atoms

    def think(self, request: BrainRequest, *, asleep: bool = True, reason: str | None = None) -> Thought:
        atoms = self.pick_atoms(request.message)
        if asleep and self.sleepy_atom and self.sleepy_atom not in atoms:
            atoms.append(self.sleepy_atom)
        atoms = atoms[:MAX_INTENT_ATOMS]
        mood = self.sleepy_atom if (asleep and self.sleepy_atom) else (request.mood or DEFAULT_MOOD)
        return Thought(intent=Intent(tuple(atoms)), mood=mood, source="local", asleep=asleep, reason=reason)
