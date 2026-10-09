"""Shared value types passed between the brain, the glyph model, the bot and training.

These are plain, immutable, JSON-serialisable records so they can be stored in
a database row or sent between processes without dragging torch along.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from . import concepts

MAX_INTENT_ATOMS = 3
GLYPH_FORMAT_VERSION = 1


@dataclass(frozen=True)
class Intent:
    """What Skeebert means: 1-3 distinct concept atoms.

    The atoms are stored in a canonical (alphabetical) order, so
    ``Intent(("question", "food")) == Intent(("food", "question"))``. Order is
    meaningless on purpose: the speaker treats the intent as a set, and two
    orderings of the same meaning must never produce two different glyphs or
    two different rows of training data.
    """

    concepts: tuple[str, ...]

    def __post_init__(self) -> None:
        raw = self.concepts
        if isinstance(raw, str):
            raw = (raw,)
        atoms = tuple(raw)
        if not 1 <= len(atoms) <= MAX_INTENT_ATOMS:
            raise ValueError(f"an intent needs 1-{MAX_INTENT_ATOMS} atoms, got {len(atoms)}")
        for name in atoms:
            if not isinstance(name, str) or not concepts.is_known(name):
                raise ValueError(f"unknown concept atom {name!r}")
        if len(set(atoms)) != len(atoms):
            raise ValueError(f"duplicate atoms in intent {atoms!r}")
        object.__setattr__(self, "concepts", tuple(sorted(atoms)))

    @classmethod
    def of(cls, *names: str) -> "Intent":
        return cls(tuple(names))

    @classmethod
    def from_ids(cls, ids: Iterable[int]) -> "Intent":
        return cls(tuple(concepts.by_id(int(i)).name for i in ids))

    def ids(self) -> tuple[int, ...]:
        return tuple(concepts.by_name(n).id for n in self.concepts)

    def gloss(self) -> str:
        """Human-readable meaning, e.g. ``"food; asking a question"``.

        This exact string is what scoring and the align stage embed as "the
        meaning", so it is built only from the atoms' glosses.
        """
        return "; ".join(concepts.by_name(n).gloss for n in self.concepts)

    def __str__(self) -> str:
        return "+".join(self.concepts)

    def to_json(self) -> str:
        return json.dumps(list(self.concepts))

    @classmethod
    def from_json(cls, text: str) -> "Intent":
        return cls(tuple(json.loads(text)))


@dataclass(frozen=True)
class Stroke:
    """One quadratic Bezier stroke, in the constrained space of ``render.py``."""

    x0: float
    y0: float
    x1: float
    y1: float
    x2: float
    y2: float
    half_width: float
    intensity: float

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1, self.x2, self.y2, self.half_width, self.intensity]

    @classmethod
    def from_list(cls, values: Sequence[float]) -> "Stroke":
        if len(values) != 8:
            raise ValueError(f"a stroke has 8 numbers, got {len(values)}")
        vals = [float(v) for v in values]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("stroke values must be finite")
        return cls(*vals)


@dataclass(frozen=True)
class Glyph:
    """A picture Skeebert drew: its stroke parameters plus which model drew it.

    The strokes are the source of truth -- the image is always re-rendered
    from them -- so training can later re-render the exact glyph a human saw
    at whatever resolution it needs. ``model_version`` ties the glyph to the
    checkpoint that produced it (see ``glyph.py``); ``variation`` is the
    optional noise seed passed to ``GlyphEngine.speak``.
    """

    strokes: tuple[Stroke, ...]
    model_version: str
    variation: int | None = None

    def __post_init__(self) -> None:
        if not self.strokes:
            raise ValueError("a glyph needs at least one stroke")
        if not self.model_version:
            raise ValueError("a glyph must record the model_version that drew it")
        object.__setattr__(self, "strokes", tuple(self.strokes))

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": GLYPH_FORMAT_VERSION,
            "model_version": self.model_version,
            "variation": self.variation,
            "strokes": [s.as_list() for s in self.strokes],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict) -> "Glyph":
        fmt = data.get("format", GLYPH_FORMAT_VERSION)
        if fmt != GLYPH_FORMAT_VERSION:
            raise ValueError(f"unsupported glyph format {fmt}")
        variation = data.get("variation")
        return cls(
            strokes=tuple(Stroke.from_list(s) for s in data["strokes"]),
            model_version=str(data["model_version"]),
            variation=None if variation is None else int(variation),
        )

    @classmethod
    def from_json(cls, text: str) -> "Glyph":
        return cls.from_dict(json.loads(text))

    # -- tensors & images ----------------------------------------------------
    def to_tensor(self):
        """[K, 8] float32 tensor of stroke parameters."""
        import torch

        return torch.tensor([s.as_list() for s in self.strokes], dtype=torch.float32)

    def coverage(self, size: int):
        """[size, size] ink coverage tensor, exactly what the model sees at that size."""
        import torch

        from .render import render

        with torch.no_grad():
            return render(self.to_tensor(), size)

    def render(self, size: int = 512):
        """The glyph as a coloured PIL RGB image (what Discord users see)."""
        from .render import colorize

        return colorize(self.coverage(size))

    def png_bytes(self, size: int = 512) -> bytes:
        buf = io.BytesIO()
        self.render(size).save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def jpeg_bytes(self, size: int = 512, quality: int = 92) -> bytes:
        buf = io.BytesIO()
        self.render(size).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


@dataclass(frozen=True)
class TrainingExample:
    """One human reading of one glyph.

    ``glyph`` is the exact glyph that was shown, ``guess_text`` what the human
    typed, ``score`` the ``scoring.score_guess`` result in [0, 1].
    """

    intent: Intent
    glyph: Glyph
    guess_text: str
    score: float

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.score) <= 1.0) or math.isnan(float(self.score)):
            raise ValueError(f"score must be in [0, 1], got {self.score}")
        object.__setattr__(self, "score", float(self.score))

    def to_dict(self) -> dict:
        return {
            "intent": list(self.intent.concepts),
            "glyph": self.glyph.to_dict(),
            "guess_text": self.guess_text,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingExample":
        return cls(
            intent=Intent(tuple(data["intent"])),
            glyph=Glyph.from_dict(data["glyph"]),
            guess_text=str(data["guess_text"]),
            score=float(data["score"]),
        )
