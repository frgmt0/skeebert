"""Score a human's guess against what Skeebert actually meant.

The formula
-----------
Let ``g`` be the embedding of the guess, ``m`` the embedding of the whole
intent gloss (``Intent.gloss()``), and ``a_1..a_n`` the embeddings of each
atom's own gloss. All are unit vectors, so dot products are cosines.

    whole    = g . m
    coverage = mean_i (g . a_i)
    raw      = 0.5 * whole + 0.5 * coverage
    score    = clip((raw - FLOOR) / (CEIL - FLOOR), 0, 1)

Why both terms:

* ``whole`` rewards a guess that captures the combined meaning ("are you
  hungry?" for food+question) even when it shares no single word with any one
  atom.
* ``coverage`` gives partial credit atom by atom. If the intent is
  food+question and the guess is just "food", the guess matches one of the two
  atoms well and the other poorly, so coverage lands near the middle -- more
  than an unrelated guess, less than a full one. That ordering is the signal
  that lets individual glyph parts become meaningful (tested in
  ``tests/test_scoring.py``).

For a single-atom intent both terms are the same comparison, so the score is
just a rescaled cosine.

Why the rescale: sentence-embedding cosines are never near 0 for "unrelated"
or near 1 for "same meaning, different words". For all-MiniLM-L6-v2, unrelated
short phrases typically sit around 0.0-0.2 and decent paraphrases around
0.6-0.9. FLOOR and CEIL stretch that useful band onto [0, 1] so a score reads
like "how right was this". They are calibration constants chosen from the
model's known behaviour, not fitted to Skeebert's human data (there is none
yet); revisit them once real guesses exist.
"""

from __future__ import annotations

import numpy as np

from . import concepts
from .embed import TextEmbedder, l2_normalize
from .types import Intent

FLOOR = 0.15
CEIL = 0.80
WHOLE_WEIGHT = 0.5


def score_guess(intent: Intent, guess_text: str, embedder: TextEmbedder) -> float:
    """How well ``guess_text`` matches ``intent``, in [0, 1]. Blank guesses score 0."""
    guess = (guess_text or "").strip()
    if not guess:
        return 0.0
    atom_glosses = [concepts.by_name(n).gloss for n in intent.concepts]
    vecs = l2_normalize(embedder.encode([guess, intent.gloss(), *atom_glosses]))
    g, whole_vec, atom_vecs = vecs[0], vecs[1], vecs[2:]
    whole = float(np.dot(g, whole_vec))
    coverage = float(np.mean(atom_vecs @ g))
    raw = WHOLE_WEIGHT * whole + (1.0 - WHOLE_WEIGHT) * coverage
    return rescale(raw)


def rescale(raw_cosine: float) -> float:
    return float(np.clip((raw_cosine - FLOOR) / (CEIL - FLOOR), 0.0, 1.0))
