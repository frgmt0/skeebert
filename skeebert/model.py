"""Skeebert's three small networks, trained from scratch.

* ``Speaker``    intent atoms -> K stroke parameters (the "mouth").
* ``Listener``   glyph image  -> one logit per atom (a machine reader, used to
                 bootstrap a language before any human has seen a glyph).
* ``HumanProxy`` glyph image  -> predicted sentence embedding of what a human
                 would guess (fit to real Discord guesses, then used to pull
                 the speaker toward glyphs humans actually read correctly).

Design notes
------------
The speaker treats an intent as a *set*: atom embeddings go through a
transformer encoder with no positional encoding, then K learned "stroke slot"
queries cross-attend to them through a transformer decoder, one stroke per
slot. Two consequences we want:

1. Order never matters (food+question == question+food).
2. Each stroke slot can learn to attend mostly to one atom, so the glyph for
   food+question can literally contain the food strokes plus the question
   strokes. That is the architectural half of compositionality; the training
   pressures in ``train.py`` are the other half.

The listener and proxy share an architecture (``GlyphEncoder``): a small
conv net whose final 4x4 feature map is flattened rather than globally pooled,
so *where* a stroke sits on the canvas stays visible to the classifier.

All three are sized by ``ModelConfig``; the default totals ~20M parameters
(see ``ModelConfig.default`` and docs/ML.md for exact counts).
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import concepts
from .render import COORD_LIMIT, MAX_HALF_WIDTH, MIN_HALF_WIDTH, PARAMS_PER_STROKE
from .types import MAX_INTENT_ATOMS, Intent

CHECKPOINT_FORMAT = 1


@dataclass(frozen=True)
class ModelConfig:
    n_concepts: int = len(concepts.CONCEPTS)
    n_strokes: int = 6
    # speaker
    d_model: int = 384
    n_heads: int = 6
    enc_layers: int = 2
    dec_layers: int = 3
    ff_mult: int = 4
    # listener / proxy image encoders
    image_size: int = 64
    channels: tuple[int, ...] = (48, 96, 192, 320)
    hidden: int = 512
    embed_dim: int = 384  # must equal the sentence embedder's dim
    # rendering at train time
    render_segments: int = 12
    # scale of the speaker's latent noise when GlyphEngine.speak(variation=...)
    variation_scale: float = 0.35

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", tuple(self.channels))
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.image_size % (2 ** len(self.channels)):
            raise ValueError("image_size must be divisible by 2**len(channels)")

    @classmethod
    def default(cls) -> "ModelConfig":
        return cls()

    @classmethod
    def tiny(cls) -> "ModelConfig":
        """A few-hundred-thousand-parameter config for tests and smoke runs."""
        return cls(
            n_strokes=3, d_model=32, n_heads=2, enc_layers=1, dec_layers=1, ff_mult=2,
            image_size=32, channels=(8, 16), hidden=32, render_segments=6,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channels"] = list(self.channels)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: (tuple(v) if k == "channels" else v) for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def pack_intents(intents: Sequence[Intent], device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Intents -> (ids [B, 3] long, mask [B, 3] bool, True where an atom is present)."""
    ids = torch.zeros(len(intents), MAX_INTENT_ATOMS, dtype=torch.long)
    mask = torch.zeros(len(intents), MAX_INTENT_ATOMS, dtype=torch.bool)
    for i, intent in enumerate(intents):
        atom_ids = intent.ids()
        ids[i, : len(atom_ids)] = torch.tensor(atom_ids)
        mask[i, : len(atom_ids)] = True
    return ids.to(device), mask.to(device)


def multi_hot(ids: torch.Tensor, mask: torch.Tensor, n_concepts: int) -> torch.Tensor:
    """(ids, mask) -> [B, n_concepts] float 0/1 target."""
    out = torch.zeros(ids.shape[0], n_concepts, device=ids.device)
    out.scatter_(1, ids, mask.float())
    return out


def constrain_strokes(raw: torch.Tensor) -> torch.Tensor:
    """Map unconstrained [..., 8] network outputs into the renderer's space."""
    pts = torch.tanh(raw[..., :6]) * COORD_LIMIT
    half_w = MIN_HALF_WIDTH + (MAX_HALF_WIDTH - MIN_HALF_WIDTH) * torch.sigmoid(raw[..., 6:7])
    ink = torch.sigmoid(raw[..., 7:8])
    return torch.cat([pts, half_w, ink], dim=-1)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# networks
# ---------------------------------------------------------------------------

class Speaker(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.atom_emb = nn.Embedding(cfg.n_concepts, d)
        enc_layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, d * cfg.ff_mult, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.enc_layers, enable_nested_tensor=False)
        dec_layer = nn.TransformerDecoderLayer(
            d, cfg.n_heads, d * cfg.ff_mult, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.dec_layers)
        self.stroke_queries = nn.Parameter(torch.randn(cfg.n_strokes, d) * 0.5)
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, PARAMS_PER_STROKE)
        # Per-slot default stroke: gives each slot its own home position and
        # shape on the canvas, so the K strokes of a glyph start out distinct
        # instead of all being near-copies of one atom-driven stroke.
        self.slot_bias = nn.Parameter(torch.randn(cfg.n_strokes, PARAMS_PER_STROKE) * 0.9)
        nn.init.normal_(self.atom_emb.weight, std=0.5)
        # Spread untrained strokes across the canvas (raw coords ~ N(0, 1)
        # before tanh) and start them mostly visible, so even the random-init
        # glyphs are varied, legible drawings rather than a dot in the middle.
        nn.init.normal_(self.head.weight, std=0.8 / d**0.5)
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[6] = -0.5
            self.head.bias[7] = 1.5

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """ids/mask [B, 3] -> constrained strokes [B, K, 8]. ``noise`` [B, K, d] is optional."""
        atoms = self.atom_emb(ids)
        pad = ~mask
        memory = self.encoder(atoms, src_key_padding_mask=pad)
        queries = self.stroke_queries.unsqueeze(0).expand(ids.shape[0], -1, -1)
        if noise is not None:
            queries = queries + noise
        h = self.decoder(queries, memory, memory_key_padding_mask=pad)
        return constrain_strokes(self.head(self.out_norm(h)) + self.slot_bias)

    def grow(self, n_concepts: int) -> None:
        """Add embedding rows for atoms appended to the vocabulary since this model was built."""
        old = self.atom_emb.weight.shape[0]
        if n_concepts <= old:
            return
        new = nn.Embedding(n_concepts, self.atom_emb.weight.shape[1]).to(self.atom_emb.weight.device)
        with torch.no_grad():
            nn.init.normal_(new.weight, std=0.5)
            new.weight[:old] = self.atom_emb.weight
        self.atom_emb = new


class GlyphEncoder(nn.Module):
    """Conv stack: [B, 1, S, S] -> [B, hidden]. Shared design for listener and proxy."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 1
        for c in cfg.channels:
            layers += [
                nn.Conv2d(c_in, c, 3, padding=1),
                nn.GroupNorm(min(8, c), c),
                nn.GELU(),
                nn.Conv2d(c, c, 3, padding=1),
                nn.GroupNorm(min(8, c), c),
                nn.GELU(),
                nn.AvgPool2d(2),
            ]
            c_in = c
        self.convs = nn.Sequential(*layers)
        side = cfg.image_size // (2 ** len(cfg.channels))
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(c_in * side * side, cfg.hidden), nn.GELU())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 3:
            images = images.unsqueeze(1)
        # Centre ink coverage around zero so an empty canvas is not "all signal".
        return self.proj(self.convs(images * 2.0 - 1.0))


class Listener(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = GlyphEncoder(cfg)
        self.mlp = nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden), nn.GELU())
        self.out = nn.Linear(cfg.hidden, cfg.n_concepts)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """[B, S, S] coverage -> [B, n_concepts] independent (multi-label) logits."""
        return self.out(self.mlp(self.encoder(images)))

    def grow(self, n_concepts: int) -> None:
        old = self.out.out_features
        if n_concepts <= old:
            return
        new = nn.Linear(self.out.in_features, n_concepts).to(self.out.weight.device)
        with torch.no_grad():
            new.weight[:old] = self.out.weight
            new.bias[:old] = self.out.bias
        self.out = new


class HumanProxy(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = GlyphEncoder(cfg)
        self.mlp = nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, cfg.embed_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """[B, S, S] coverage -> [B, embed_dim] unit vectors (predicted guess embedding)."""
        return F.normalize(self.mlp(self.encoder(images)), dim=-1)


# ---------------------------------------------------------------------------
# checkpoint bundle
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    """Everything one checkpoint file holds.

    ``meta`` records provenance: which stage produced it, its parent version,
    whether the proxy has been fit to human data, and when.
    """

    config: ModelConfig
    speaker: Speaker
    listener: Listener
    proxy: HumanProxy
    meta: dict = field(default_factory=dict)

    @classmethod
    def create(cls, config: ModelConfig, seed: int) -> "ModelBundle":
        """Deterministic random init: same config + seed -> same weights on the same torch build."""
        gen_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(seed)
            speaker = Speaker(config)
            listener = Listener(config)
            proxy = HumanProxy(config)
        finally:
            torch.random.set_rng_state(gen_state)
        return cls(config, speaker, listener, proxy, meta={"stage": "init", "seed": seed, "parent": None, "proxy_trained": False})

    def to(self, device: torch.device | str) -> "ModelBundle":
        self.speaker.to(device)
        self.listener.to(device)
        self.proxy.to(device)
        return self

    def param_counts(self) -> dict[str, int]:
        s, l, p = count_params(self.speaker), count_params(self.listener), count_params(self.proxy)
        return {"speaker": s, "listener": l, "proxy": p, "total": s + l + p}

    def state(self) -> dict:
        return {
            "speaker": {k: v.detach().cpu() for k, v in self.speaker.state_dict().items()},
            "listener": {k: v.detach().cpu() for k, v in self.listener.state_dict().items()},
            "proxy": {k: v.detach().cpu() for k, v in self.proxy.state_dict().items()},
        }

    def version(self) -> str:
        """Content hash of config + all weights (first 12 hex chars of sha256).

        This is the ``model_version`` stamped on every glyph. Hashing the
        weights rather than counting runs means a version always names exactly
        one set of numbers, and two machines that produce the same weights agree.
        """
        h = hashlib.sha256()
        h.update(json.dumps(self.config.to_dict(), sort_keys=True).encode())
        for part, sd in sorted(self.state().items()):
            for key in sorted(sd):
                t = sd[key].contiguous()
                h.update(f"{part}.{key}:{tuple(t.shape)}:{t.dtype}".encode())
                h.update(t.numpy().tobytes())
        return h.hexdigest()[:12]

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        torch.save(
            {"format": CHECKPOINT_FORMAT, "config": self.config.to_dict(), "meta": dict(self.meta), **self.state()},
            buf,
        )
        return buf.getvalue()

    @classmethod
    def from_file(cls, path, device: torch.device | str = "cpu") -> "ModelBundle":
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path}: unsupported checkpoint format {data.get('format')}")
        cfg = ModelConfig.from_dict(data["config"])
        bundle = cls(cfg, Speaker(cfg), Listener(cfg), HumanProxy(cfg), meta=dict(data.get("meta") or {}))
        bundle.speaker.load_state_dict(data["speaker"])
        bundle.listener.load_state_dict(data["listener"])
        bundle.proxy.load_state_dict(data["proxy"])
        return bundle.to(device)

    def grow_concepts(self, n_concepts: int) -> bool:
        """Extend atom tables to ``n_concepts`` (append-only vocab). Returns True if anything grew."""
        if n_concepts <= self.config.n_concepts:
            return False
        self.speaker.grow(n_concepts)
        self.listener.grow(n_concepts)
        self.config = ModelConfig.from_dict({**self.config.to_dict(), "n_concepts": n_concepts})
        self.speaker.cfg = self.listener.cfg = self.proxy.cfg = self.config
        return True
