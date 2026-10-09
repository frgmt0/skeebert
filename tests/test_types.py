import io

import pytest
from PIL import Image

from skeebert.types import Glyph, Intent, Stroke, TrainingExample


def test_intent_is_canonical_and_order_free():
    a = Intent(("question", "food"))
    b = Intent.of("food", "question")
    assert a == b and hash(a) == hash(b)
    assert a.concepts == ("food", "question")
    assert str(a) == "food+question"


def test_intent_accepts_single_string():
    assert Intent("food").concepts == ("food",)


@pytest.mark.parametrize("bad", [(), ("food", "water", "fire", "ice"), ("food", "food"), ("not-an-atom",)])
def test_intent_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        Intent(bad)


def test_intent_ids_gloss_and_json():
    i = Intent.of("food", "question")
    assert Intent.from_ids(i.ids()) == i
    assert i.gloss() == "food; asking a question"
    assert Intent.from_json(i.to_json()) == i


def _glyph():
    return Glyph(
        strokes=(
            Stroke(-0.5, -0.5, 0.0, 0.3, 0.5, -0.4, 0.03, 0.9),
            Stroke(0.2, 0.6, 0.4, 0.0, -0.3, 0.2, 0.045, 0.7),
        ),
        model_version="abc123def456",
        variation=7,
    )


def test_glyph_json_round_trip():
    g = _glyph()
    assert Glyph.from_json(g.to_json()) == g


def test_glyph_requires_version_and_strokes():
    with pytest.raises(ValueError):
        Glyph(strokes=(), model_version="x")
    with pytest.raises(ValueError):
        Glyph(strokes=_glyph().strokes, model_version="")
    with pytest.raises(ValueError):
        Stroke.from_list([0.0] * 7)
    with pytest.raises(ValueError):
        Stroke.from_list([float("nan")] * 8)


def test_glyph_png_and_jpeg_bytes_are_valid_images():
    g = _glyph()
    png = g.png_bytes(128)
    jpg = g.jpeg_bytes(128)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert jpg[:3] == b"\xff\xd8\xff"
    for data, fmt in ((png, "PNG"), (jpg, "JPEG")):
        im = Image.open(io.BytesIO(data))
        assert im.format == fmt and im.size == (128, 128) and im.mode == "RGB"


def test_rendered_glyph_has_ink_where_strokes_are():
    im = _glyph().render(64)
    px = im.load()
    # canvas (0,0) maps to pixel centre; first stroke passes near its control polygon
    background = px[0, 0]
    assert max(sum(px[x, y]) for x in range(64) for y in range(64)) > sum(background) + 300


def test_training_example_round_trip_and_validation():
    ex = TrainingExample(Intent.of("food"), _glyph(), "hungry?", 0.75)
    assert TrainingExample.from_dict(ex.to_dict()) == ex
    with pytest.raises(ValueError):
        TrainingExample(Intent.of("food"), _glyph(), "x", 1.5)
