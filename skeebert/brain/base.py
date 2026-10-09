"""What a brain is: something that turns a message into what Skeebert means.

A brain never sees or picks glyph shapes. It only chooses an ``Intent`` (1-3
atoms from ``skeebert.concepts``) and Skeebert's mood afterwards (one atom from
the ``feeling`` category). The glyph engine (the mouth) draws the intent.

Two brains exist:

* ``haiku.HaikuBrain``: Claude Haiku via Anthropic's API (the awake brain).
* ``local.LocalBrain``: nearest atoms by sentence embedding (the sleeping
  brain), used when there is no API key, when the daily budget is spent,
  when a person hits the hourly rate limit, or when the API call fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence

from .. import concepts
from ..types import Intent

BrainSource = Literal["haiku", "local"]

DEFAULT_MOOD = "curious"


def feeling_atoms(vocabulary: Sequence[str] | None = None) -> list[str]:
    allowed = set(vocabulary) if vocabulary is not None else None
    return [c.name for c in concepts.CONCEPTS if c.category == "feeling" and (allowed is None or c.name in allowed)]


@dataclass(frozen=True)
class Turn:
    """One earlier Skeebert-addressed message in the same conversation, and what Skeebert meant back."""

    text: str
    reply: Intent | None = None


@dataclass(frozen=True)
class BrainRequest:
    message: str
    mood: str = DEFAULT_MOOD
    history: tuple[Turn, ...] = field(default_factory=tuple)
    user_key: str | None = None  # pseudonym, used only for the per-user rate limit


@dataclass(frozen=True)
class Thought:
    intent: Intent
    mood: str
    source: BrainSource
    asleep: bool
    reason: str | None = None  # why it fell asleep: no_api_key | budget | rate_limited | api_error | bad_output


class Brain(Protocol):
    def think(self, request: BrainRequest) -> Thought: ...
