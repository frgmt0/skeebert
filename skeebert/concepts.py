"""The fixed vocabulary of concept atoms Skeebert can mean.

The LLM brain expresses what Skeebert wants to say as an *intent*: one to
three of these atoms. Skeebert's own model then has to invent a glyph for
that intent. The atoms are deliberately coarse and few (~200) -- a small,
closed set is what makes it possible for a from-scratch model to build a
shared visual vocabulary with humans at all, and for repeated use to make
individual glyph parts recognisable.

THE APPEND-ONLY RULE
--------------------
An atom's id is its position in ``_ATOMS``. Checkpoints size their embedding
tables by atom id, and every stored intent, glyph and human guess refers to
atoms by id or name. So:

* never reorder, rename or delete an entry;
* only ever append new atoms at the end.

Breaking this silently re-labels every glyph the model has learned and every
row of collected training data. ``tests/test_concepts.py`` pins a fingerprint
of the existing names so a reorder or deletion fails the test suite, while
appending still passes.

Glosses are short natural phrases rather than dictionary definitions because
they are embedded with a sentence model and compared against what humans type
as guesses ("he's hungry!"), so they should read like something a person
would say.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Concept:
    id: int
    name: str
    category: str
    gloss: str


# (name, category, gloss). Position == id. APPEND ONLY -- see module docstring.
_ATOMS: tuple[tuple[str, str, str], ...] = (
    # --- who ---------------------------------------------------------------
    ("self", "people", "me, myself"),
    ("you", "people", "you, the person I'm talking to"),
    ("we", "people", "us, all of us together"),
    ("they", "people", "them, other people"),
    ("friend", "people", "a friend"),
    ("stranger", "people", "a stranger, someone unknown"),
    ("human", "people", "a human being"),
    ("everyone", "people", "everyone, all people"),
    ("family", "people", "family"),
    ("creature", "people", "a creature or animal"),
    # --- feelings ------------------------------------------------------------
    ("happy", "feeling", "feeling happy"),
    ("sad", "feeling", "feeling sad"),
    ("angry", "feeling", "feeling angry"),
    ("afraid", "feeling", "feeling scared"),
    ("excited", "feeling", "feeling excited"),
    ("bored", "feeling", "feeling bored"),
    ("tired", "feeling", "feeling tired, sleepy"),
    ("curious", "feeling", "feeling curious, wondering"),
    ("confused", "feeling", "feeling confused"),
    ("surprised", "feeling", "feeling surprised"),
    ("calm", "feeling", "feeling calm and peaceful"),
    ("lonely", "feeling", "feeling lonely"),
    ("proud", "feeling", "feeling proud"),
    ("embarrassed", "feeling", "feeling embarrassed"),
    ("grateful", "feeling", "feeling thankful"),
    ("love", "feeling", "love, loving someone"),
    ("hate", "feeling", "hate, really disliking"),
    ("worried", "feeling", "feeling worried, anxious"),
    ("hopeful", "feeling", "feeling hopeful"),
    ("amused", "feeling", "finding something funny"),
    ("disgusted", "feeling", "feeling disgusted, gross"),
    ("jealous", "feeling", "feeling jealous"),
    ("hungry", "feeling", "feeling hungry"),
    ("thirsty", "feeling", "feeling thirsty"),
    ("pain", "feeling", "pain, it hurts"),
    ("comfortable", "feeling", "feeling cozy and comfortable"),
    ("sick", "feeling", "feeling sick, unwell"),
    ("brave", "feeling", "feeling brave"),
    # --- speech acts -----------------------------------------------------
    ("question", "speech", "asking a question"),
    ("yes", "speech", "yes, agreeing"),
    ("no", "speech", "no, disagreeing"),
    ("greeting", "speech", "hello, greeting"),
    ("goodbye", "speech", "goodbye, leaving"),
    ("thanks", "speech", "thank you"),
    ("sorry", "speech", "sorry, apologizing"),
    ("please", "speech", "please, asking nicely"),
    ("maybe", "speech", "maybe, not sure"),
    ("warning", "speech", "warning, watch out"),
    ("request", "speech", "asking for something"),
    ("offer", "speech", "offering to give something"),
    ("agree", "speech", "I agree"),
    ("disagree", "speech", "I disagree"),
    ("joke", "speech", "telling a joke"),
    ("compliment", "speech", "a compliment, praising"),
    ("insult", "speech", "an insult, teasing"),
    ("promise", "speech", "a promise"),
    ("story", "speech", "telling a story"),
    ("name", "speech", "a name, what something is called"),
    ("why", "speech", "why, asking the reason"),
    ("how", "speech", "how, asking the way"),
    ("where", "speech", "where, asking the place"),
    ("when", "speech", "when, asking the time"),
    ("what", "speech", "what, asking which thing"),
    ("who", "speech", "who, asking which person"),
    # --- actions -------------------------------------------------------------
    ("eat", "action", "eating"),
    ("drink", "action", "drinking"),
    ("sleep", "action", "sleeping"),
    ("play", "action", "playing"),
    ("work", "action", "working"),
    ("make", "action", "making or building something"),
    ("break", "action", "breaking something"),
    ("fix", "action", "fixing, repairing"),
    ("go", "action", "going, moving away"),
    ("come", "action", "coming here"),
    ("stay", "action", "staying, waiting here"),
    ("see", "action", "seeing, looking"),
    ("hear", "action", "hearing, listening"),
    ("speak", "action", "speaking, talking"),
    ("think", "action", "thinking"),
    ("know", "action", "knowing"),
    ("learn", "action", "learning"),
    ("teach", "action", "teaching"),
    ("help", "action", "helping"),
    ("give", "action", "giving"),
    ("take", "action", "taking"),
    ("want", "action", "wanting something"),
    ("need", "action", "needing something"),
    ("like", "action", "liking something"),
    ("find", "action", "finding something"),
    ("lose", "action", "losing something"),
    ("win", "action", "winning"),
    ("fight", "action", "fighting"),
    ("run", "action", "running"),
    ("fly", "action", "flying"),
    ("swim", "action", "swimming"),
    ("sing", "action", "singing"),
    ("dance", "action", "dancing"),
    ("wait", "action", "waiting"),
    ("remember", "action", "remembering"),
    ("forget", "action", "forgetting"),
    ("try", "action", "trying"),
    ("share", "action", "sharing"),
    ("hide", "action", "hiding"),
    ("search", "action", "searching, looking for"),
    ("open", "action", "opening"),
    ("close", "action", "closing, shutting"),
    ("grow", "action", "growing"),
    ("die", "action", "dying"),
    ("live", "action", "living, being alive"),
    ("save", "action", "saving, rescuing"),
    ("count", "action", "counting numbers"),
    # --- things --------------------------------------------------------------
    ("food", "thing", "food"),
    ("water", "thing", "water"),
    ("home", "thing", "home, house"),
    ("ship", "thing", "a ship, spaceship"),
    ("machine", "thing", "a machine"),
    ("tool", "thing", "a tool"),
    ("gift", "thing", "a gift, present"),
    ("music", "thing", "music"),
    ("game", "thing", "a game"),
    ("book", "thing", "a book"),
    ("picture", "thing", "a picture, drawing"),
    ("word", "thing", "a word, language"),
    ("light", "thing", "light, brightness"),
    ("dark", "thing", "darkness"),
    ("fire", "thing", "fire, heat"),
    ("ice", "thing", "ice, cold"),
    ("air", "thing", "air, breathing"),
    ("rock", "thing", "a rock, stone"),
    ("metal", "thing", "metal"),
    ("plant", "thing", "a plant, tree"),
    ("animal", "thing", "an animal, pet"),
    ("money", "thing", "money"),
    ("science", "thing", "science, experiments"),
    ("energy", "thing", "energy, power"),
    ("problem", "thing", "a problem, trouble"),
    ("idea", "thing", "an idea"),
    ("secret", "thing", "a secret"),
    ("dream", "thing", "a dream"),
    ("body", "thing", "a body"),
    ("hand", "thing", "a hand"),
    ("eye", "thing", "an eye"),
    ("heart", "thing", "a heart"),
    ("message", "thing", "a message"),
    # --- nature & places -------------------------------------------------------
    ("sun", "place", "the sun, a star"),
    ("star", "place", "the stars"),
    ("planet", "place", "a planet, a world"),
    ("earth", "place", "Earth"),
    ("space", "place", "outer space"),
    ("sky", "place", "the sky"),
    ("sea", "place", "the sea, ocean"),
    ("mountain", "place", "a mountain"),
    ("forest", "place", "a forest"),
    ("city", "place", "a city"),
    ("room", "place", "a room"),
    ("here", "place", "here, this place"),
    ("there", "place", "there, that place"),
    ("far", "place", "far away"),
    ("near", "place", "near, close by"),
    ("inside", "place", "inside"),
    ("outside", "place", "outside"),
    ("up", "place", "up, above"),
    ("down", "place", "down, below"),
    ("weather", "place", "the weather"),
    ("rain", "place", "rain"),
    # --- time ------------------------------------------------------------------
    ("now", "time", "now, right now"),
    ("before", "time", "before, in the past"),
    ("after", "time", "after, later"),
    ("today", "time", "today"),
    ("tomorrow", "time", "tomorrow"),
    ("yesterday", "time", "yesterday"),
    ("morning", "time", "morning"),
    ("night", "time", "night time"),
    ("soon", "time", "soon"),
    ("always", "time", "always, forever"),
    ("never", "time", "never"),
    ("again", "time", "again, one more time"),
    ("fast", "time", "fast, quickly"),
    ("slow", "time", "slow, slowly"),
    # --- quantities ----------------------------------------------------------
    ("one", "quantity", "one, a single thing"),
    ("two", "quantity", "two, a pair"),
    ("three", "quantity", "three"),
    ("many", "quantity", "many, lots"),
    ("few", "quantity", "a few, a little"),
    ("all", "quantity", "all of it"),
    ("none", "quantity", "none, nothing"),
    ("more", "quantity", "more"),
    ("less", "quantity", "less"),
    ("half", "quantity", "half"),
    ("enough", "quantity", "enough"),
    # --- modifiers & qualities ----------------------------------------------
    ("not", "modifier", "not, the opposite"),
    ("very", "modifier", "very, extremely"),
    ("big", "modifier", "big, large"),
    ("small", "modifier", "small, tiny"),
    ("good", "modifier", "good"),
    ("bad", "modifier", "bad"),
    ("new", "modifier", "new"),
    ("old", "modifier", "old"),
    ("hot", "modifier", "hot"),
    ("cold", "modifier", "cold"),
    ("loud", "modifier", "loud"),
    ("quiet", "modifier", "quiet"),
    ("strong", "modifier", "strong"),
    ("weak", "modifier", "weak"),
    ("beautiful", "modifier", "beautiful, pretty"),
    ("ugly", "modifier", "ugly"),
    ("easy", "modifier", "easy"),
    ("hard", "modifier", "hard, difficult"),
    ("same", "modifier", "the same"),
    ("different", "modifier", "different"),
    ("true", "modifier", "true, real"),
    ("false", "modifier", "false, fake"),
    ("important", "modifier", "important"),
    ("weird", "modifier", "weird, strange"),
    ("safe", "modifier", "safe"),
    ("danger", "modifier", "dangerous"),
    ("with", "modifier", "together with"),
    ("without", "modifier", "without"),
)


def _build() -> tuple[Concept, ...]:
    return tuple(
        Concept(id=i, name=name, category=category, gloss=gloss)
        for i, (name, category, gloss) in enumerate(_ATOMS)
    )


CONCEPTS: tuple[Concept, ...] = _build()
_BY_NAME: dict[str, Concept] = {c.name: c for c in CONCEPTS}
if len(_BY_NAME) != len(CONCEPTS):  # pragma: no cover - guarded by tests too
    raise RuntimeError("duplicate concept names in skeebert.concepts._ATOMS")

CATEGORIES: tuple[str, ...] = tuple(dict.fromkeys(c.category for c in CONCEPTS))


def by_name(name: str) -> Concept:
    """Look up an atom by name. Raises KeyError with a helpful message."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown concept atom {name!r}") from None


def by_id(concept_id: int) -> Concept:
    if not 0 <= concept_id < len(CONCEPTS):
        raise KeyError(f"unknown concept id {concept_id}")
    return CONCEPTS[concept_id]


def names() -> list[str]:
    """All atom names in id order."""
    return [c.name for c in CONCEPTS]


def is_known(name: str) -> bool:
    return name in _BY_NAME
