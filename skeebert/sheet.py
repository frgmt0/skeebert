"""Contact sheet: up to nine glyphs on one image, numbered 1-9, for ``/skeebert dictionary``.

Each glyph is re-rendered from its stored strokes (the same picture people
saw), laid out three to a row on the glyph background colour, with its number
in the top-left corner so the text list below the image can refer to it.
"""

from __future__ import annotations

import io
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from .types import Glyph

MAX_GLYPHS = 9
COLUMNS = 3
CELL = 256
GAP = 8
BACKGROUND = (8, 10, 20)
LABEL_FILL = (255, 196, 110)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


def contact_sheet(glyphs: Sequence[Glyph], cell: int = CELL) -> Image.Image:
    if not glyphs:
        raise ValueError("a contact sheet needs at least one glyph")
    if len(glyphs) > MAX_GLYPHS:
        raise ValueError(f"at most {MAX_GLYPHS} glyphs fit on a contact sheet, got {len(glyphs)}")
    cols = min(COLUMNS, len(glyphs))
    rows = (len(glyphs) + COLUMNS - 1) // COLUMNS
    width = cols * cell + (cols + 1) * GAP
    height = rows * cell + (rows + 1) * GAP
    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = _font(max(14, cell // 8))
    for i, glyph in enumerate(glyphs):
        r, c = divmod(i, COLUMNS)
        x = GAP + c * (cell + GAP)
        y = GAP + r * (cell + GAP)
        sheet.paste(glyph.render(cell), (x, y))
        draw.text((x + 8, y + 4), str(i + 1), fill=LABEL_FILL, font=font)
    return sheet


def contact_sheet_png(glyphs: Sequence[Glyph], cell: int = CELL) -> bytes:
    buf = io.BytesIO()
    contact_sheet(glyphs, cell).save(buf, format="PNG", optimize=True)
    return buf.getvalue()
