"""Export a curated set of REAL Skeebert glyphs for the public site.

Every glyph here is drawn by ``GlyphEngine`` from whatever checkpoint the given
directory serves (``current`` if set, else ``init.pt``). Nothing is hand-drawn
or retouched. For each intent it writes:

* ``<slug>.png``  -- the exact 512x512 image Discord users see (``Glyph.png_bytes``)
* ``<slug>.webp`` / ``<slug>-thumb.webp`` -- the same renderer at web sizes

plus ``glyphs.json``, a manifest the site reads at runtime: model version,
a human label (``--label``), whether the served file is the untrained
``init.pt``, real parameter counts from the checkpoint, and each glyph's
intent, gloss, files and raw stroke parameters (so the browser can redraw the
strokes as SVG exactly).

Usage::

    uv run python scripts/render_site_glyphs.py --checkpoint-dir checkpoints \
        --out site/public/glyphs --label "trained, model <version>"

The script never creates or modifies anything in the checkpoint directory: it
refuses to run on a directory that has neither ``current`` nor ``init.pt``
(where ``GlyphEngine.load`` would otherwise create a fresh ``init.pt``).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from skeebert import concepts
from skeebert.brain.haiku import PERSONA_EXAMPLES
from skeebert.glyph import CURRENT_NAME, INIT_NAME, GlyphEngine
from skeebert.render import render as render_coverage, colorize
from skeebert.types import MAX_INTENT_ATOMS, Intent

# The curated set, in display order. The first entry is the hero glyph.
CURATED: tuple[tuple[str, ...], ...] = (
    ("greeting",),
    ("food", "question"),
    ("happy", "you"),
    ("music",),
    ("rain",),
    ("sad", "why"),
    ("animal", "excited", "question"),
    ("star",),
    ("love",),
    ("sleep",),
    ("fire",),
    ("friend",),
)

EXAMPLE_SIZE = 384  # the persona-example strip

WEB_SIZE = 768  # the decode stage shows these at up to ~560 CSS px
THUMB_SIZE = 160  # the picker strip
PNG_SIZE = 512


def _slug(intent: Intent) -> str:
    return "-".join(intent.concepts)


def _webp_bytes(glyph, size: int) -> bytes:
    import torch

    with torch.no_grad():
        cov = render_coverage(glyph.to_tensor(), size)
    buf = io.BytesIO()
    colorize(cov).save(buf, format="WEBP", quality=86, method=6)
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="output directory, e.g. site/public/glyphs")
    ap.add_argument(
        "--label",
        default=None,
        help='how the site names the model that drew the glyphs, e.g. "untrained (random weights), model <version>". '
        "'{version}' is replaced with the model version. Default is derived from the checkpoint.",
    )
    args = ap.parse_args(argv)

    ck: Path = args.checkpoint_dir
    if not ((ck / CURRENT_NAME).exists() or (ck / INIT_NAME).exists()):
        print(f"error: {ck} has neither {CURRENT_NAME} nor {INIT_NAME}; refusing to create a model", file=sys.stderr)
        return 2

    engine = GlyphEngine.load(ck)
    version = engine.model_version
    untrained = engine.path.name == INIT_NAME
    if args.label:
        label = args.label.replace("{version}", version)
    else:
        label = f"untrained (random weights), model {version}" if untrained else f"model {version}"

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    # Remove only files a previous run of this script wrote (listed in its manifest).
    old_manifest = out / "glyphs.json"
    if old_manifest.exists():
        try:
            old = json.loads(old_manifest.read_text())
            for g in old.get("glyphs", []) + old.get("examples", []):
                for key in ("png", "webp", "thumb"):
                    name = g.get(key)
                    if name and "/" not in name and (out / name).exists():
                        (out / name).unlink()
        except (ValueError, OSError):
            pass

    glyphs = []
    for atoms in CURATED:
        intent = Intent(atoms)
        glyph = engine.speak(intent)
        slug = _slug(intent)
        (out / f"{slug}.png").write_bytes(glyph.png_bytes(PNG_SIZE))
        (out / f"{slug}.webp").write_bytes(_webp_bytes(glyph, WEB_SIZE))
        (out / f"{slug}-thumb.webp").write_bytes(_webp_bytes(glyph, THUMB_SIZE))
        glyphs.append(
            {
                "id": slug,
                "intent": list(atoms),  # display order as curated; the model treats it as a set
                "gloss": intent.gloss(),
                "png": f"{slug}.png",
                "webp": f"{slug}.webp",
                "thumb": f"{slug}-thumb.webp",
                "model_version": glyph.model_version,
                "strokes": [s.as_list() for s in glyph.strokes],
            }
        )

    # The persona's worked examples (situation -> atoms, from skeebert/brain/haiku.py),
    # drawn by the same model. The site shows them as a constructed exchange, labelled as such.
    examples = []
    for situation, atoms in PERSONA_EXAMPLES:
        intent = Intent(atoms)
        glyph = engine.speak(intent)
        slug = "ex-" + _slug(intent)
        (out / f"{slug}.webp").write_bytes(_webp_bytes(glyph, EXAMPLE_SIZE))
        examples.append(
            {
                "id": slug,
                "situation": situation,
                "intent": list(atoms),  # the persona's order (most important first)
                "gloss": intent.gloss(),
                "webp": f"{slug}.webp",
                "model_version": glyph.model_version,
                "strokes": [s.as_list() for s in glyph.strokes],
            }
        )

    counts = engine.bundle.param_counts()
    cfg = engine.bundle.config
    n = cfg.n_concepts
    categories = sorted({c.category for c in concepts.CONCEPTS[:n]})
    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": version,
        "checkpoint_file": engine.path.name,
        "untrained": untrained,
        "label": label,
        "png_size": PNG_SIZE,
        "web_size": WEB_SIZE,
        "thumb_size": THUMB_SIZE,
        "model": {
            "params": counts,
            "n_strokes": cfg.n_strokes,
            "params_per_stroke": 8,
            "n_concepts": n,
            "n_categories": len(categories),
            "max_atoms": MAX_INTENT_ATOMS,
            "possible_messages": sum(math.comb(n, k) for k in range(1, MAX_INTENT_ATOMS + 1)),
        },
        "glyphs": glyphs,
        "examples_source": "skeebert/brain/haiku.py PERSONA_EXAMPLES",
        "examples": examples,
    }
    old_manifest.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    print(f"wrote {len(glyphs)} glyphs and {len(examples)} persona examples from {engine.path.name} ({label}) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
