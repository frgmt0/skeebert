import hashlib

from skeebert import concepts

# Fingerprint of the first FROZEN_COUNT atom names in id order. Appending new
# atoms keeps this passing; reordering, renaming or deleting any existing atom
# breaks it -- which is the point (see the append-only rule in concepts.py).
# Only ever bump FROZEN_COUNT/FROZEN_SHA together *after* appending, never to
# paper over an edit to existing entries.
FROZEN_COUNT = 218
FROZEN_SHA = "f5429e39e31789aba0c8ecee962218a6a195c78c6a9c19b62cc8ed02e72a3048"


def _fingerprint(n):
    return hashlib.sha256("\n".join(concepts.names()[:n]).encode()).hexdigest()


def test_roughly_two_hundred_atoms():
    assert 180 <= len(concepts.CONCEPTS) <= 260


def test_ids_are_positions_and_names_unique():
    assert [c.id for c in concepts.CONCEPTS] == list(range(len(concepts.CONCEPTS)))
    names = concepts.names()
    assert len(set(names)) == len(names)
    assert all(n and n == n.strip() and n.islower() for n in names)


def test_every_atom_has_category_and_gloss():
    for c in concepts.CONCEPTS:
        assert c.category in concepts.CATEGORIES
        assert c.gloss.strip()


def test_categories_cover_the_planned_kinds():
    for cat in ("people", "feeling", "speech", "action", "thing", "place", "time", "quantity", "modifier"):
        assert cat in concepts.CATEGORIES


def test_append_only_fingerprint():
    assert len(concepts.CONCEPTS) >= FROZEN_COUNT
    assert _fingerprint(FROZEN_COUNT) == FROZEN_SHA


def test_lookup_helpers():
    food = concepts.by_name("food")
    assert concepts.by_id(food.id) is food
    assert concepts.is_known("question") and not concepts.is_known("banana-phone")
    try:
        concepts.by_name("nope")
    except KeyError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("expected KeyError")
