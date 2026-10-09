"""Differentiable stroke rasterizer -- the one and only way a glyph becomes pixels.

A glyph is K quadratic Bezier strokes. Each stroke is 8 numbers::

    x0, y0, x1, y1, x2, y2, half_width, intensity

Points live in the canvas square [-1, 1] x [-1, 1] (x right, y down),
``half_width`` is in the same units, and ``intensity`` in [0, 1] says how much
ink the stroke lays down (a stroke with intensity ~0 is effectively absent,
which lets the model use fewer than K strokes).

Why a soft distance field: the speaker is trained by backpropagating a
listener's loss *through the picture*, so the rasterizer must have useful
gradients with respect to every stroke parameter. Each pixel's ink is a
sigmoid of (half_width - distance to the stroke), so moving a control point or
thickening a line changes nearby pixels smoothly.

Why the SAME function at 64px and 512px: training sees 64x64 images, Discord
users see 512x512 ones. If export used a different renderer, humans would be
judging a picture the model never saw. Here the edge softness is defined in
*pixels* (``softness_px``), so every resolution is the same continuous image,
just anti-aliased at its own pixel size; a 512 render box-downsampled to 64
matches the 64 render closely (tested).

Everything is deterministic: no sampling, no atomics-dependent reductions.
"""

from __future__ import annotations

import torch

PARAMS_PER_STROKE = 8

# Bounds the speaker maps its raw outputs into (see model.constrain_strokes).
# Kept here because they are a property of the drawing surface: below
# MIN_HALF_WIDTH a line vanishes at the 64px training resolution (one pixel is
# 2/64 = 0.031 canvas units wide), above MAX_HALF_WIDTH a single stroke
# blots out a large part of the canvas.
MIN_HALF_WIDTH = 0.016
MAX_HALF_WIDTH = 0.05
COORD_LIMIT = 0.8  # keep control points off the very edge of the canvas
# NOTE: these constants are not part of model_version. Changing them re-draws
# every existing glyph differently under the same version -- treat them as frozen
# once real glyphs have been shown to people.


def _pixel_grid(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Pixel-centre coordinates, shape [size*size, 2] as (x, y)."""
    half = 1.0 / size
    lin = torch.linspace(-1.0 + half, 1.0 - half, size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def bezier_points(strokes: torch.Tensor, segments: int) -> torch.Tensor:
    """Sample each quadratic Bezier at ``segments + 1`` evenly spaced t.

    strokes: [..., 8] -> returns [..., segments + 1, 2]
    """
    p0 = strokes[..., 0:2].unsqueeze(-2)
    p1 = strokes[..., 2:4].unsqueeze(-2)
    p2 = strokes[..., 4:6].unsqueeze(-2)
    t = torch.linspace(0.0, 1.0, segments + 1, device=strokes.device, dtype=strokes.dtype)
    t = t.view(*([1] * (strokes.dim() - 1)), -1, 1)
    u = 1.0 - t
    return u * u * p0 + 2.0 * u * t * p1 + t * t * p2


def render(
    strokes: torch.Tensor,
    size: int,
    *,
    segments: int = 12,
    softness_px: float = 0.6,
) -> torch.Tensor:
    """Rasterize strokes into an ink-coverage image.

    strokes: [B, K, 8] (or [K, 8] for a single glyph), already in the
             constrained parameter space described in the module docstring.
    size:    output is size x size.
    returns: [B, size, size] (or [size, size]) coverage in [0, 1], where 0 is
             empty canvas and 1 is full ink. Colour is applied later by
             ``colorize``; the model only ever sees this single channel.

    Strokes are combined as a soft union, 1 - prod(1 - c_k), so overlapping
    strokes do not exceed full ink and every stroke keeps a gradient.
    Memory is O(B * segments * size^2) per stroke (strokes are processed one
    at a time), which keeps a 512px render of one glyph to a few tens of MB.
    """
    single = strokes.dim() == 2
    if single:
        strokes = strokes.unsqueeze(0)
    if strokes.dim() != 3 or strokes.shape[-1] != PARAMS_PER_STROKE:
        raise ValueError(f"expected strokes [B, K, {PARAMS_PER_STROKE}], got {tuple(strokes.shape)}")
    batch, n_strokes, _ = strokes.shape
    pix = _pixel_grid(size, strokes.device, strokes.dtype)  # [P, 2]
    softness = softness_px * (2.0 / size)
    pts = bezier_points(strokes, segments)  # [B, K, S+1, 2]

    empty = torch.ones(batch, size * size, device=strokes.device, dtype=strokes.dtype)
    for k in range(n_strokes):
        a = pts[:, k, :-1, :].unsqueeze(2)  # [B, S, 1, 2]
        b = pts[:, k, 1:, :].unsqueeze(2)
        ab = b - a
        ap = pix.view(1, 1, -1, 2) - a  # [B, S, P, 2]
        denom = (ab * ab).sum(-1).clamp_min(1e-8)
        t = ((ap * ab).sum(-1) / denom).clamp(0.0, 1.0)  # [B, S, P]
        diff = ap - t.unsqueeze(-1) * ab
        d2 = (diff * diff).sum(-1).amin(dim=1)  # nearest segment, [B, P]
        dist = torch.sqrt(d2 + 1e-10)
        half_w = strokes[:, k, 6].unsqueeze(-1)
        ink = strokes[:, k, 7].unsqueeze(-1)
        cover = torch.sigmoid((half_w - dist) / softness) * ink
        empty = empty * (1.0 - cover)
    out = (1.0 - empty).view(batch, size, size)
    return out[0] if single else out


# ---------------------------------------------------------------------------
# Presentation (not differentiable, not seen by the model)
# ---------------------------------------------------------------------------

# Pale bone-white ink with a warm amber halo on a deep blue-black field: reads
# as something etched or glowing on a hull panel rather than a doodle, and
# stays legible after Discord's JPEG thumbnails.
BACKGROUND_RGB = (14, 16, 24)
VIGNETTE_RGB = (26, 30, 44)
INK_RGB = (240, 232, 214)
GLOW_RGB = (226, 150, 64)


def colorize(coverage, *, glow: bool = True):
    """Turn a [H, W] coverage array (numpy or tensor, values 0..1) into a PIL RGB image.

    The glow is a blurred copy of the ink composited underneath it. It is
    purely cosmetic and deterministic; the model is trained on raw coverage.
    """
    import numpy as np
    from PIL import Image, ImageFilter

    if isinstance(coverage, torch.Tensor):
        coverage = coverage.detach().to("cpu", torch.float32).numpy()
    cov = np.clip(np.asarray(coverage, dtype=np.float32), 0.0, 1.0)
    h, w = cov.shape

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xs + 0.5) / w - 0.5) ** 2 + ((ys + 0.5) / h - 0.5) ** 2) / 0.7071
    centre = np.clip(1.0 - r, 0.0, 1.0)[..., None] ** 1.5
    bg = np.array(BACKGROUND_RGB, np.float32) * (1 - centre) + np.array(VIGNETTE_RGB, np.float32) * centre

    img = bg
    if glow:
        radius = max(1.0, w / 48.0)
        cov_img = Image.fromarray((cov * 255).astype(np.uint8))
        halo = np.asarray(cov_img.filter(ImageFilter.GaussianBlur(radius)), np.float32) / 255.0
        halo = np.clip(halo * 0.85, 0.0, 1.0)[..., None]
        img = img * (1 - halo) + np.array(GLOW_RGB, np.float32) * halo
    c = cov[..., None]
    img = img * (1 - c) + np.array(INK_RGB, np.float32) * c
    return Image.fromarray(np.clip(img + 0.5, 0, 255).astype(np.uint8))
