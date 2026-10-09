import torch
import torch.nn.functional as F

from skeebert.model import constrain_strokes
from skeebert.render import render


def _strokes(seed=0, batch=2, k=4):
    g = torch.Generator().manual_seed(seed)
    return constrain_strokes(torch.randn(batch, k, 8, generator=g))


def test_output_shape_and_range():
    img = render(_strokes(), 64)
    assert img.shape == (2, 64, 64)
    assert img.min() >= 0 and img.max() <= 1
    assert img.mean() > 0.01  # strokes actually drew something
    single = render(_strokes()[0], 32)
    assert single.shape == (32, 32)


def test_gradients_flow_to_every_stroke_parameter():
    raw = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(1), requires_grad=True)
    img = render(constrain_strokes(raw), 48)
    target = torch.zeros_like(img)
    target[:, 10:30, 10:30] = 1
    ((img - target) ** 2).mean().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    # every one of the 8 parameter kinds receives gradient for at least one stroke
    assert (raw.grad.abs().sum(dim=(0, 1)) > 0).all()


def test_deterministic():
    s = _strokes(3)
    assert torch.equal(render(s, 64), render(s, 64))


def test_hi_res_matches_low_res_after_downsampling():
    s = _strokes(5, batch=1)
    lo = render(s, 64)
    hi = render(s, 512)
    down = F.avg_pool2d(hi.unsqueeze(1), 8).squeeze(1)
    assert (down - lo).abs().mean() < 0.02
    assert (down - lo).abs().max() < 0.35


def test_zero_intensity_stroke_draws_nothing():
    s = _strokes(2, batch=1, k=1).clone()
    s[..., 7] = 0.0
    assert render(s, 32).abs().max() == 0
