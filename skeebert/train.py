"""Training pipeline: how Skeebert's glyph language is grown. See docs/ML.md for the story.

Three stages, run separately and in order:

1. ``bootstrap`` -- self-play. The speaker draws a glyph for a random intent,
   the listener must pick that intent out of a set of distractors from the
   picture alone. No human data. This gives Skeebert a *consistent* language
   (same meaning -> recognisably same glyph parts) before anyone sees it.
2. ``proxy`` -- fit ``HumanProxy`` to real Discord data: the exact glyph a
   human was shown -> the sentence embedding of what they guessed.
3. ``align`` -- fine-tune the speaker so the proxy predicts that humans would
   guess the right meaning, with the frozen listener's game loss as a
   regulariser so the language does not dissolve into whatever fools the proxy.

Every run starts from an existing checkpoint (the served one unless ``base``
names another), writes a brand-new ``ckpt-<version>.pt`` and never touches the
``current`` pointer -- promotion is a separate, explicit act (``glyph.promote``).
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import concepts
from .embed import TextEmbedder
from .glyph import (
    INIT_NAME,
    checkpoint_path_for,
    ensure_init_checkpoint,
    save_new_checkpoint,
    serving_checkpoint_path,
)
from .model import Listener, ModelBundle, ModelConfig, multi_hot
from .render import PARAMS_PER_STROKE, render
from .types import MAX_INTENT_ATOMS, Intent, TrainingExample

STAGES = ("bootstrap", "proxy", "align")


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class TrainingRefused(RuntimeError):
    """Training will not start; the message says why and what to do."""


class TrainingNotConfirmed(TrainingRefused):
    pass


class InsufficientHumanData(TrainingRefused):
    pass


class ProxyNotTrained(TrainingRefused):
    pass


class TrainingDiverged(RuntimeError):
    """Loss went non-finite. Nothing is saved when this happens."""


# ---------------------------------------------------------------------------
# config / report
# ---------------------------------------------------------------------------

DEFAULT_STEPS = {"bootstrap": 30_000, "proxy": 3_000, "align": 5_000}


@dataclass
class TrainConfig:
    steps: int = 30_000
    batch_size: int = 64
    lr: float = 3e-4
    grad_clip: float = 1.0
    seed: int = 0
    # intent sampling: probability of a 1-, 2- and 3-atom intent
    atom_count_probs: tuple[float, float, float] = (0.4, 0.35, 0.25)
    # referential game
    n_distractors: int = 15
    hard_distractor_frac: float = 0.5
    game_weight: float = 1.0
    bce_weight: float = 1.0
    ink_weight: float = 0.05
    speaker_noise: float = 0.1
    listener_reset_every: int = 5_000  # 0 disables iterated-learning resets
    # augmentation applied to every image a listener/proxy sees in training
    aug_translate: float = 0.06
    aug_rotate_deg: float = 8.0
    aug_scale: float = 0.08
    aug_noise: float = 0.04
    # human-data stages
    min_human_examples: int = 300
    proxy_val_frac: float = 0.1
    align_game_weight: float = 0.5
    align_example_frac: float = 0.5
    # bookkeeping
    eval_size: int = 512
    log_every: int = 50
    eta_probe_steps: int = 3

    def __post_init__(self) -> None:
        self.atom_count_probs = tuple(float(p) for p in self.atom_count_probs)
        if len(self.atom_count_probs) != MAX_INTENT_ATOMS or abs(sum(self.atom_count_probs) - 1.0) > 1e-6:
            raise ValueError(f"atom_count_probs must be {MAX_INTENT_ATOMS} probabilities summing to 1")
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be positive")

    @classmethod
    def for_stage(cls, stage: str, **overrides) -> "TrainConfig":
        _check_stage(stage)
        overrides.setdefault("steps", DEFAULT_STEPS[stage])
        if stage != "bootstrap":
            overrides.setdefault("lr", 1e-4)
        return cls(**overrides)

    @classmethod
    def tiny(cls, **overrides) -> "TrainConfig":
        """A few-step config for tests and smoke runs."""
        base = dict(
            steps=3, batch_size=8, n_distractors=3, listener_reset_every=2, min_human_examples=8,
            eval_size=16, log_every=1, eta_probe_steps=1,
        )
        base.update(overrides)
        return cls(**base)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["atom_count_probs"] = list(self.atom_count_probs)
        return d


@dataclass
class TrainReport:
    stage: str
    steps: int
    device: str
    parent_version: str
    version: str
    checkpoint_path: str
    log_path: str
    seconds: float
    seconds_per_step: float
    final_loss: float
    metrics: dict = field(default_factory=dict)
    promoted: bool = False  # always False: training never promotes

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RuntimeEstimate:
    stage: str
    device: str
    steps: int
    seconds_per_step: float
    setup_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.setup_seconds + self.seconds_per_step * self.steps

    def describe(self) -> str:
        return (
            f"{self.stage} on {self.device}: ~{self.seconds_per_step:.3f}s/step x {self.steps} steps "
            f"= ~{format_duration(self.total_seconds)}"
        )


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------

def select_device(prefer: str | torch.device | None = None) -> torch.device:
    """``None``/``"auto"`` picks cuda, then mps, then cpu. An explicit name must be available."""
    if isinstance(prefer, torch.device):
        return prefer
    choice = (prefer or "auto").lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice.startswith("cuda") and not torch.cuda.is_available():
        raise TrainingRefused("device 'cuda' requested but CUDA is not available")
    if choice == "mps" and not torch.backends.mps.is_available():
        raise TrainingRefused("device 'mps' requested but MPS is not available")
    return torch.device(choice)


# ---------------------------------------------------------------------------
# sampling, augmentation, game loss
# ---------------------------------------------------------------------------

def sample_intents(
    batch: int, n_concepts: int, probs: Sequence[float], gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly random distinct atoms, 1-3 per intent. Returns (ids [B, 3], mask [B, 3]) on CPU."""
    ids = torch.rand(batch, n_concepts, generator=gen).argsort(dim=1)[:, :MAX_INTENT_ATOMS]
    counts = torch.multinomial(torch.tensor(probs, dtype=torch.float), batch, replacement=True, generator=gen) + 1
    mask = torch.arange(MAX_INTENT_ATOMS).unsqueeze(0) < counts.unsqueeze(1)
    return ids, mask


def make_distractors(
    ids: torch.Tensor, mask: torch.Tensor, n_distractors: int, n_concepts: int,
    hard_frac: float, probs: Sequence[float], gen: torch.Generator,
) -> torch.Tensor:
    """Candidate intents to confuse the target with, as multi-hot [B, D, N] (CPU).

    A *hard* distractor is the target with one atom swapped, dropped or added.
    These are what force compositional glyphs: to tell food+question from
    food+yes, the listener must be able to see the question part on its own,
    so the speaker has to draw each atom recognisably rather than drawing one
    holistic blob per whole intent. *Random* distractors keep the task honest
    about the rest of the vocabulary.
    """
    b, d, n = ids.shape[0], n_distractors, n_concepts
    target = multi_hot(ids, mask, n)
    hard = target.unsqueeze(1).repeat(1, d, 1)
    pos = (torch.rand(b, d, MAX_INTENT_ATOMS, generator=gen) * mask.unsqueeze(1)).argmax(-1)
    old = ids.unsqueeze(1).expand(b, d, MAX_INTENT_ATOMS).gather(-1, pos.unsqueeze(-1))  # [B, D, 1]
    new = torch.randint(0, n, (b, d, 1), generator=gen)
    op = torch.rand(b, d, generator=gen)
    count = mask.sum(-1, keepdim=True)  # [B, 1]
    drop = (op >= 0.6) & (op < 0.8) & (count > 1)
    add = (op >= 0.8) & (count < MAX_INTENT_ATOMS)
    replace = ~(drop | add)
    remove_old = (replace | drop).float().unsqueeze(-1)
    hard.scatter_add_(-1, old, -remove_old)
    add_new = (replace | add).float().unsqueeze(-1)
    hard.scatter_(-1, new, torch.maximum(hard.gather(-1, new), add_new))

    r_ids, r_mask = sample_intents(b * d, n, probs, gen)
    rand = multi_hot(r_ids, r_mask, n).view(b, d, n)
    use_hard = (torch.rand(b, d, 1, generator=gen) < hard_frac).float()
    return use_hard * hard + (1.0 - use_hard) * rand


def game_loss(logits: torch.Tensor, target: torch.Tensor, candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Referential-game cross-entropy, multi-label BCE, and candidate accuracy.

    Each candidate intent is scored by its log-likelihood under the listener's
    independent per-atom Bernoulli outputs. Terms for atoms where candidates
    agree cancel, so the comparison is decided exactly by the atoms that
    differ -- which is the point of hard distractors. Candidates identical to
    the target (possible by chance) are masked out.
    """
    lp, ln = F.logsigmoid(logits), F.logsigmoid(-logits)
    delta = lp - ln  # [B, N]
    tgt_score = (target * delta).sum(-1, keepdim=True)
    cand_score = torch.einsum("bdn,bn->bd", candidates, delta)
    same = (candidates == target.unsqueeze(1)).all(-1)
    cand_score = cand_score.masked_fill(same, float("-inf"))
    scores = torch.cat([tgt_score, cand_score], dim=1)
    zeros = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    ce = F.cross_entropy(scores, zeros)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum(-1).mean()
    acc = (scores.argmax(-1) == 0).float().mean()
    return ce, bce, acc


def augment(images: torch.Tensor, cfg: TrainConfig, gen: torch.Generator) -> torch.Tensor:
    """Random small rotation/scale/shift plus pixel noise; differentiable w.r.t. ``images``.

    Humans will see glyphs at different sizes, through JPEG, on phones. A
    language that only works pixel-perfectly is useless to them, and
    augmentation pushes the speaker toward bold, shape-level distinctions.
    """
    b = images.shape[0]
    dev = images.device
    ang = (torch.rand(b, generator=gen) * 2 - 1) * math.radians(cfg.aug_rotate_deg)
    scale = 1.0 + (torch.rand(b, generator=gen) * 2 - 1) * cfg.aug_scale
    shift = (torch.rand(b, 2, generator=gen) * 2 - 1) * cfg.aug_translate
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], -1), torch.stack([sin, cos, shift[:, 1]], -1)], 1)
    theta = theta.to(dev, images.dtype)
    x = images.unsqueeze(1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(1)
    if cfg.aug_noise > 0:
        out = out + torch.randn(out.shape, generator=gen).to(dev, out.dtype) * cfg.aug_noise
    return out.clamp(0.0, 1.0)


def topographic_similarity(target: torch.Tensor, images: torch.Tensor) -> float:
    """Spearman correlation between meaning distance and glyph distance over all pairs.

    Meaning distance is the Jaccard distance between atom sets, glyph distance
    the L2 distance between coverage images. A compositional language scores
    high: intents that share atoms get glyphs that share strokes. Random
    glyphs score near 0. This is the standard "topsim" measure from the
    emergent-communication literature.
    """
    t = target.float().cpu()
    inter = t @ t.T
    union = t.sum(1, keepdim=True) + t.sum(1) - inter
    mdist = 1.0 - inter / union.clamp_min(1.0)
    flat = images.reshape(images.shape[0], -1).float().cpu()
    gdist = torch.cdist(flat, flat)
    iu = torch.triu_indices(t.shape[0], t.shape[0], offset=1)
    a, b = mdist[iu[0], iu[1]], gdist[iu[0], iu[1]]
    if a.numel() < 2:
        return 0.0
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = float(ra.norm() * rb.norm())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Average ranks (ties share the mean rank), as Spearman requires."""
    vals, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True)
    ends = counts.cumsum(0).float()
    avg = ends - (counts.float() - 1) / 2.0
    return avg[inverse]


def _intent_from_row(ids: torch.Tensor, mask: torch.Tensor) -> Intent:
    return Intent.from_ids(int(i) for i, m in zip(ids.tolist(), mask.tolist()) if m)


def _glyph_tensor(example: TrainingExample, n_strokes: int) -> torch.Tensor:
    """The shown glyph's strokes, padded with invisible strokes if it came from a smaller model."""
    t = example.glyph.to_tensor()
    if t.shape[0] < n_strokes:
        pad = torch.zeros(n_strokes - t.shape[0], PARAMS_PER_STROKE)
        pad[:, 6] = 0.02  # zero intensity means no ink; width just needs to be valid
        t = torch.cat([t, pad], 0)
    return t


# ---------------------------------------------------------------------------
# stage runners
# ---------------------------------------------------------------------------

class _Stage:
    """One training stage bound to a bundle. ``step()`` does one optimiser update."""

    def __init__(self, bundle: ModelBundle, cfg: TrainConfig, device: torch.device):
        self.bundle, self.cfg, self.device = bundle, cfg, device
        self.mcfg: ModelConfig = bundle.config
        self.gen = torch.Generator().manual_seed(cfg.seed)

    def _render(self, strokes: torch.Tensor) -> torch.Tensor:
        return render(strokes, self.mcfg.image_size, segments=self.mcfg.render_segments)

    def _game(self, ids, mask, images, listener) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        n = self.mcfg.n_concepts
        target = multi_hot(ids, mask, n)
        cands = make_distractors(
            ids.cpu(), mask.cpu(), cfg.n_distractors, n, cfg.hard_distractor_frac, cfg.atom_count_probs, self.gen
        )
        logits = listener(augment(images, cfg, self.gen))
        return game_loss(logits, target, cands.to(self.device))

    def _clip(self, params) -> None:
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)

    def _eval_game(self) -> dict:
        """Clean (un-augmented) listener accuracy and topsim on a fixed set of intents."""
        cfg, n = self.cfg, self.mcfg.n_concepts
        gen = torch.Generator().manual_seed(10_000 + cfg.seed)
        ids, mask = sample_intents(cfg.eval_size, n, cfg.atom_count_probs, gen)
        sp, li = self.bundle.speaker, self.bundle.listener
        was = sp.training, li.training
        sp.eval(), li.eval()
        accs, exact, imgs = [], [], []
        with torch.no_grad():
            for s in range(0, ids.shape[0], cfg.batch_size):
                bi, bm = ids[s : s + cfg.batch_size].to(self.device), mask[s : s + cfg.batch_size].to(self.device)
                im = self._render(sp(bi, bm))
                logits = li(im)
                target = multi_hot(bi, bm, n)
                cands = make_distractors(bi.cpu(), bm.cpu(), cfg.n_distractors, n, cfg.hard_distractor_frac,
                                         cfg.atom_count_probs, gen).to(self.device)
                _, _, acc = game_loss(logits, target, cands)
                accs.append(acc.item() * bi.shape[0])
                exact.append(((logits > 0).float() == target).all(-1).float().sum().item())
                imgs.append(im.cpu())
        sp.train(was[0]), li.train(was[1])
        images = torch.cat(imgs)
        return {
            "eval_candidate_acc": sum(accs) / ids.shape[0],
            "eval_exact_set_acc": sum(exact) / ids.shape[0],
            "eval_topsim": topographic_similarity(multi_hot(ids, mask, n), images),
            "eval_ink": float(images.mean()),
        }


class _Bootstrap(_Stage):
    def __init__(self, bundle, cfg, device, examples, embedder):
        super().__init__(bundle, cfg, device)
        bundle.speaker.train()
        bundle.listener.train()
        self.opt_s = torch.optim.Adam(bundle.speaker.parameters(), lr=cfg.lr)
        self.opt_l = torch.optim.Adam(bundle.listener.parameters(), lr=cfg.lr)
        self.resets = 0

    def _maybe_reset_listener(self, step: int) -> bool:
        """Iterated learning: periodically replace the listener with a fresh one.

        A new listener must learn the speaker's language from scratch, quickly.
        Languages built from reusable parts are faster to learn, so over many
        generations the speaker is pushed toward compositional, systematic
        glyphs (Ren et al. 2020, "Compositional languages emerge in a neural
        iterated learning model"). No reset happens in the final interval, so
        the saved listener is a competent reader of the saved speaker.
        """
        every = self.cfg.listener_reset_every
        if every <= 0 or step == 0 or step % every or step + every > self.cfg.steps:
            return False
        self.resets += 1
        torch.manual_seed(self.cfg.seed * 1000 + self.resets)
        fresh = Listener(self.mcfg).to(self.device)
        self.bundle.listener.load_state_dict(fresh.state_dict())
        self.opt_l = torch.optim.Adam(self.bundle.listener.parameters(), lr=self.cfg.lr)
        return True

    def step(self, step: int) -> dict:
        cfg, mcfg = self.cfg, self.mcfg
        reset = self._maybe_reset_listener(step)
        ids, mask = sample_intents(cfg.batch_size, mcfg.n_concepts, cfg.atom_count_probs, self.gen)
        noise = None
        if cfg.speaker_noise > 0:
            noise = (torch.randn(cfg.batch_size, mcfg.n_strokes, mcfg.d_model, generator=self.gen) * cfg.speaker_noise).to(self.device)
        ids_d, mask_d = ids.to(self.device), mask.to(self.device)
        images = self._render(self.bundle.speaker(ids_d, mask_d, noise))
        ce, bce, acc = self._game(ids_d, mask_d, images, self.bundle.listener)
        ink = images.mean()
        loss = cfg.game_weight * ce + cfg.bce_weight * bce + cfg.ink_weight * ink
        self.opt_s.zero_grad(set_to_none=True)
        self.opt_l.zero_grad(set_to_none=True)
        loss.backward()
        self._clip(list(self.bundle.speaker.parameters()) + list(self.bundle.listener.parameters()))
        self.opt_s.step()
        self.opt_l.step()
        return {"loss": loss.item(), "game_ce": ce.item(), "bce": bce.item(), "acc": acc.item(),
                "ink": ink.item(), "listener_reset": int(reset)}

    def evaluate(self) -> dict:
        return {**self._eval_game(), "listener_resets": self.resets}

    def meta_updates(self) -> dict:
        return {}


class _Proxy(_Stage):
    def __init__(self, bundle, cfg, device, examples, embedder):
        super().__init__(bundle, cfg, device)
        usable = [e for e in examples if e.guess_text.strip()]
        self.n_examples = len(usable)
        vecs = np.asarray(embedder.encode([e.guess_text.strip() for e in usable]), dtype=np.float32)
        if vecs.shape != (len(usable), self.mcfg.embed_dim):
            raise ValueError(f"embedder returned {vecs.shape}, expected ({len(usable)}, {self.mcfg.embed_dim})")
        self.targets = F.normalize(torch.from_numpy(vecs), dim=-1).to(device)
        strokes = torch.stack([_glyph_tensor(e, self.mcfg.n_strokes) for e in usable])
        with torch.no_grad():
            imgs = [render(strokes[s : s + 256], self.mcfg.image_size, segments=self.mcfg.render_segments)
                    for s in range(0, len(usable), 256)]
        self.images = torch.cat(imgs).to(device)
        perm = torch.randperm(len(usable), generator=torch.Generator().manual_seed(cfg.seed))
        n_val = int(len(usable) * cfg.proxy_val_frac) if len(usable) >= 10 else 0
        self.val_idx, self.train_idx = perm[:n_val], perm[n_val:]
        bundle.proxy.train()
        self.opt = torch.optim.Adam(bundle.proxy.parameters(), lr=cfg.lr)

    def step(self, step: int) -> dict:
        pick = torch.randint(0, len(self.train_idx), (self.cfg.batch_size,), generator=self.gen)
        idx = self.train_idx[pick].to(self.device)
        pred = self.bundle.proxy(augment(self.images[idx], self.cfg, self.gen))
        cos = (pred * self.targets[idx]).sum(-1)
        loss = (1.0 - cos).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self._clip(self.bundle.proxy.parameters())
        self.opt.step()
        return {"loss": loss.item(), "cos": cos.mean().item()}

    def _cos(self, idx: torch.Tensor) -> float:
        if len(idx) == 0:
            return float("nan")
        self.bundle.proxy.eval()
        with torch.no_grad():
            idx = idx.to(self.device)
            pred = self.bundle.proxy(self.images[idx])
            out = float((pred * self.targets[idx]).sum(-1).mean())
        self.bundle.proxy.train()
        return out

    def evaluate(self) -> dict:
        """Proxy cosine on train and held-out data, next to a constant-prediction baseline.

        The baseline predicts the mean training guess for every glyph. A proxy
        that does not beat it has learned "people usually say X", not anything
        about the pictures, and aligning to it would be pointless.
        """
        mean_vec = F.normalize(self.targets[self.train_idx.to(self.device)].mean(0), dim=-1)
        out = {"train_cos": self._cos(self.train_idx), "n_examples": self.n_examples,
               "n_val": int(len(self.val_idx))}
        if len(self.val_idx):
            out["val_cos"] = self._cos(self.val_idx)
            out["val_baseline_cos"] = float((self.targets[self.val_idx.to(self.device)] @ mean_vec).mean())
        return out

    def meta_updates(self) -> dict:
        return {"proxy_trained": True, "proxy_examples": self.n_examples}


class _Align(_Stage):
    def __init__(self, bundle, cfg, device, examples, embedder):
        super().__init__(bundle, cfg, device)
        self.embedder = embedder
        self.example_intents = sorted({e.intent for e in examples if e.guess_text.strip()}, key=str)
        self.n_examples = sum(1 for e in examples if e.guess_text.strip())
        self._cache: dict[Intent, torch.Tensor] = {}
        self._embed(self.example_intents)
        for m in (bundle.proxy, bundle.listener):
            m.eval()
            m.requires_grad_(False)
        bundle.speaker.train()
        self.opt = torch.optim.Adam(bundle.speaker.parameters(), lr=cfg.lr)

    def _embed(self, intents: Sequence[Intent]) -> torch.Tensor:
        missing = [i for i in dict.fromkeys(intents) if i not in self._cache]
        if missing:
            vecs = np.asarray(self.embedder.encode([i.gloss() for i in missing]), dtype=np.float32)
            if vecs.shape != (len(missing), self.mcfg.embed_dim):
                raise ValueError(f"embedder returned {vecs.shape}, expected ({len(missing)}, {self.mcfg.embed_dim})")
            for intent, v in zip(missing, F.normalize(torch.from_numpy(vecs), dim=-1)):
                self._cache[intent] = v
        return torch.stack([self._cache[i] for i in intents]).to(self.device)

    def _batch(self, size: int, gen: torch.Generator) -> list[Intent]:
        n_ex = int(round(size * self.cfg.align_example_frac)) if self.example_intents else 0
        picks = torch.randint(0, max(1, len(self.example_intents)), (n_ex,), generator=gen).tolist()
        out = [self.example_intents[i] for i in picks]
        ids, mask = sample_intents(size - n_ex, self.mcfg.n_concepts, self.cfg.atom_count_probs, gen)
        out += [_intent_from_row(i, m) for i, m in zip(ids, mask)]
        return out

    def _ids(self, intents: Sequence[Intent]) -> tuple[torch.Tensor, torch.Tensor]:
        from .model import pack_intents

        return pack_intents(intents, self.device)

    def step(self, step: int) -> dict:
        cfg = self.cfg
        intents = self._batch(cfg.batch_size, self.gen)
        target = self._embed(intents)
        ids, mask = self._ids(intents)
        images = self._render(self.bundle.speaker(ids, mask))
        pred = self.bundle.proxy(augment(images, cfg, self.gen))
        cos = (pred * target).sum(-1)
        align = (1.0 - cos).mean()
        ce, bce, acc = self._game(ids, mask, images, self.bundle.listener)
        ink = images.mean()
        loss = align + cfg.align_game_weight * (cfg.game_weight * ce + cfg.bce_weight * bce) + cfg.ink_weight * ink
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self._clip(self.bundle.speaker.parameters())
        self.opt.step()
        return {"loss": loss.item(), "align_cos": cos.mean().item(), "game_ce": ce.item(), "acc": acc.item(),
                "ink": ink.item()}

    def evaluate(self) -> dict:
        gen = torch.Generator().manual_seed(20_000 + self.cfg.seed)
        intents = self._batch(self.cfg.eval_size, gen)
        sp = self.bundle.speaker
        sp.eval()
        cos = []
        with torch.no_grad():
            for s in range(0, len(intents), self.cfg.batch_size):
                chunk = intents[s : s + self.cfg.batch_size]
                ids, mask = self._ids(chunk)
                pred = self.bundle.proxy(self._render(sp(ids, mask)))
                cos.append((pred * self._embed(chunk)).sum(-1).cpu())
        sp.train()
        return {"eval_align_cos": float(torch.cat(cos).mean()), **self._eval_game()}

    def meta_updates(self) -> dict:
        return {"align_examples": self.n_examples}


_RUNNERS = {"bootstrap": _Bootstrap, "proxy": _Proxy, "align": _Align}


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def _check_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {', '.join(STAGES)}")


def _load_base(checkpoint_dir: Path, base: str | None, model_config: ModelConfig | None, create: bool):
    if base:
        path = checkpoint_path_for(checkpoint_dir, base)
    else:
        if create:
            ensure_init_checkpoint(checkpoint_dir, model_config)
        path = serving_checkpoint_path(checkpoint_dir)
        if not path.exists():
            # estimate-only path on an empty directory: an in-memory init model
            from .glyph import INIT_SEED

            bundle = ModelBundle.create(model_config or ModelConfig.default(), INIT_SEED)
            return bundle, None
    return ModelBundle.from_file(path), path


def check_preconditions(stage: str, examples: Sequence[TrainingExample], bundle: ModelBundle, config: TrainConfig) -> None:
    """Raise ``TrainingRefused`` if ``stage`` cannot meaningfully run on this data/checkpoint."""
    _check_stage(stage)
    if stage == "bootstrap":
        return
    usable = sum(1 for e in examples if e.guess_text.strip())
    if usable < config.min_human_examples:
        raise InsufficientHumanData(
            f"stage {stage!r} needs at least {config.min_human_examples} human guesses with text, "
            f"have {usable}. Collect more guesses (or lower min_human_examples deliberately)."
        )
    if stage == "align" and not bundle.meta.get("proxy_trained"):
        raise ProxyNotTrained(
            "stage 'align' needs a checkpoint whose HumanProxy was fit to human data; "
            "run the 'proxy' stage first and pass its version as base"
        )


def _default_embedder(stage: str, embedder: TextEmbedder | None) -> TextEmbedder | None:
    if embedder is not None or stage == "bootstrap":
        return embedder
    from .embed import Embedder

    return Embedder()


def estimate_runtime(
    examples: Iterable[TrainingExample],
    stage: str,
    config: TrainConfig | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    device: str | torch.device | None = None,
    *,
    base: str | None = None,
    embedder: TextEmbedder | None = None,
    model_config: ModelConfig | None = None,
    probe_steps: int | None = None,
) -> RuntimeEstimate:
    """Time a few real optimiser steps on a throwaway copy and extrapolate. Writes nothing."""
    _check_stage(stage)
    config = config or TrainConfig.for_stage(stage)
    examples = list(examples)
    dev = select_device(device)
    t0 = time.perf_counter()
    bundle, _ = _load_base(Path(checkpoint_dir), base, model_config, create=False)
    check_preconditions(stage, examples, bundle, config)
    bundle.grow_concepts(len(concepts.CONCEPTS))
    bundle.to(dev)
    runner = _RUNNERS[stage](bundle, config, dev, examples, _default_embedder(stage, embedder))
    runner.step(0)  # warm-up: first step pays for allocator / kernel setup
    _sync(dev)
    setup = time.perf_counter() - t0
    n = max(1, probe_steps if probe_steps is not None else config.eta_probe_steps)
    t1 = time.perf_counter()
    for i in range(1, n + 1):
        runner.step(i)
    _sync(dev)
    return RuntimeEstimate(stage, str(dev), config.steps, (time.perf_counter() - t1) / n, setup)


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def run_training(
    examples: Iterable[TrainingExample],
    stage: str,
    config: TrainConfig | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    device: str | torch.device | None = None,
    *,
    base: str | None = None,
    embedder: TextEmbedder | None = None,
    model_config: ModelConfig | None = None,
    progress: Callable[[dict], None] | None = None,
) -> TrainReport:
    """Run one stage and write a new checkpoint. Never changes what is being served.

    examples:      human ``TrainingExample``s (ignored by bootstrap; required by proxy/align).
    base:          version to start from; default is the served checkpoint (``current`` or init.pt).
    embedder:      text embedder for proxy/align; default loads all-MiniLM-L6-v2.
    model_config:  only used if init.pt has to be created in an empty directory.
    progress:      optional callback receiving each logged record (for a CLI progress line).
    """
    _check_stage(stage)
    config = config or TrainConfig.for_stage(stage)
    examples = list(examples)
    dev = select_device(device)
    ckdir = Path(checkpoint_dir)
    bundle, base_path = _load_base(ckdir, base, model_config, create=True)
    parent = bundle.meta.get("version") or bundle.version()
    check_preconditions(stage, examples, bundle, config)
    grew = bundle.grow_concepts(len(concepts.CONCEPTS))
    bundle.to(dev)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    runner = _RUNNERS[stage](bundle, config, dev, examples, _default_embedder(stage, embedder))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_dir = ckdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stamp}-{stage}-{parent}.jsonl"

    t_start = time.perf_counter()
    last = {}
    with open(log_path, "x") as log:
        def emit(rec: dict) -> None:
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if progress:
                progress(rec)

        emit({"event": "start", "stage": stage, "parent": parent, "base_file": base_path.name if base_path else INIT_NAME,
              "device": str(dev), "steps": config.steps, "params": bundle.param_counts(),
              "grew_concepts": grew, "config": config.to_dict()})
        probe = max(1, config.eta_probe_steps)
        t_loop = time.perf_counter()
        for step in range(config.steps):
            last = runner.step(step)
            if not math.isfinite(last["loss"]):
                emit({"event": "diverged", "step": step, **last})
                raise TrainingDiverged(f"{stage} loss became {last['loss']} at step {step}; nothing was saved")
            if step + 1 == probe:
                _sync(dev)
                per = (time.perf_counter() - t_loop) / probe
                emit({"event": "eta", "step": step + 1, "seconds_per_step": per,
                      "eta_seconds": per * (config.steps - probe), "eta": format_duration(per * (config.steps - probe))})
            if (step + 1) % config.log_every == 0 or step + 1 == config.steps:
                emit({"event": "step", "step": step + 1, "elapsed": time.perf_counter() - t_start, **last})
        _sync(dev)
        train_seconds = time.perf_counter() - t_loop
        metrics = runner.evaluate()

        bundle.meta = {
            **{k: v for k, v in bundle.meta.items() if k in ("proxy_trained", "proxy_examples", "seed")},
            **runner.meta_updates(),
            "stage": stage,
            "parent": parent,
            "steps": config.steps,
            "created": datetime.now(timezone.utc).isoformat(),
            "train_config": config.to_dict(),
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        bundle.meta.setdefault("proxy_trained", False)
        bundle.to("cpu")
        version, path = save_new_checkpoint(bundle, ckdir)
        emit({"event": "done", "version": version, "checkpoint": path.name, "metrics": metrics,
              "seconds": time.perf_counter() - t_start})

    return TrainReport(
        stage=stage, steps=config.steps, device=str(dev), parent_version=parent, version=version,
        checkpoint_path=str(path), log_path=str(log_path), seconds=time.perf_counter() - t_start,
        seconds_per_step=train_seconds / config.steps, final_loss=float(last["loss"]), metrics=metrics,
    )


def train_from_cli(
    examples: Iterable[TrainingExample],
    stage: str,
    *,
    confirm: bool,
    config: TrainConfig | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    device: str | torch.device | None = None,
    base: str | None = None,
    embedder: TextEmbedder | None = None,
    progress: Callable[[dict], None] | None = None,
) -> TrainReport:
    """CLI entry: refuses to start a (potentially hours-long) run unless ``confirm`` is True.

    The intended CLI flow is: call ``estimate_runtime`` and show
    ``RuntimeEstimate.describe()``, then only call this with ``confirm=True``
    when the user passed an explicit flag (e.g. ``--yes``).
    """
    _check_stage(stage)
    if confirm is not True:
        raise TrainingNotConfirmed(
            f"refusing to start {stage!r} training without explicit confirmation; "
            "check the runtime estimate first, then re-run with the confirm flag"
        )
    return run_training(examples, stage, config, checkpoint_dir, device, base=base, embedder=embedder, progress=progress)
