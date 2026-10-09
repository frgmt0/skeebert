from skeebert.scoring import rescale, score_guess
from skeebert.types import Intent


def test_partial_credit_ordering(embedder):
    intent = Intent.of("food", "question")
    full = score_guess(intent, "food asking a question", embedder)
    half = score_guess(intent, "food", embedder)
    other_half = score_guess(intent, "asking a question", embedder)
    wrong = score_guess(intent, "purple elephant dancing", embedder)
    assert full > half > wrong
    assert full > other_half > wrong
    assert 0.0 <= wrong <= half <= full <= 1.0


def test_single_atom_exact_is_perfect(embedder):
    intent = Intent.of("hungry")
    assert score_guess(intent, "feeling hungry", embedder) == 1.0


def test_blank_guess_scores_zero(embedder):
    assert score_guess(Intent.of("food"), "   ", embedder) == 0.0


def test_rescale_clips():
    assert rescale(-1.0) == 0.0 and rescale(2.0) == 1.0
    assert 0.0 < rescale(0.5) < 1.0
