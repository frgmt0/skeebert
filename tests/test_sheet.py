import io

import pytest
from PIL import Image

from skeebert.sheet import contact_sheet, contact_sheet_png
from skeebert.types import Intent


def test_contact_sheet_layout(engine):
    glyphs = [engine.speak(Intent.of(a)) for a in ["food", "happy", "sad", "water", "fire"]]
    img = contact_sheet(glyphs, cell=64)
    assert img.size == (3 * 64 + 4 * 8, 2 * 64 + 3 * 8)
    png = contact_sheet_png(glyphs[:1], cell=64)
    assert Image.open(io.BytesIO(png)).size == (64 + 16, 64 + 16)


def test_contact_sheet_limits(engine):
    with pytest.raises(ValueError):
        contact_sheet([])
    with pytest.raises(ValueError):
        contact_sheet([engine.speak(Intent.of("food"))] * 10)
