"""Serving glyphs, and the rules for which checkpoint is being served.

Checkpoint directory layout
---------------------------
::

    checkpoints/
      init.pt               seeded random-init model, created once, never rewritten
      ckpt-<version>.pt     every trained model; one file per version, never overwritten
      current               text file naming the served checkpoint file (absent => init.pt)
      logs/*.jsonl          per-run training loss logs (train.py)

Rules:

* Training always writes a NEW ``ckpt-<version>.pt`` and never touches
  ``current``. A run can therefore never change what users see.
* ``promote(dir, version)`` is the only thing that changes ``current``, and it
  is an explicit, separate act (the CLI asks for it by name).
* ``init.pt`` exists so the untrained Skeebert is stable: the first time the
  engine starts with an empty directory it creates the random model from a
  fixed seed and *persists* it, so a restart serves the same glyphs and every
  stored glyph's ``model_version`` points at a file that really exists.

``model_version`` is the content hash of the weights (``ModelBundle.version``).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from . import concepts
from .model import ModelBundle, ModelConfig, pack_intents
from .types import Glyph, Intent, Stroke

INIT_NAME = "init.pt"
CURRENT_NAME = "current"
INIT_SEED = 1729  # fixed: the untrained Skeebert's "face". Changing it changes init.pt for new installs.
_DECIMALS = 5  # stroke params are stored rounded; rendering always uses the rounded values


class CheckpointError(RuntimeError):
    pass


def ckpt_name(version: str) -> str:
    return f"ckpt-{version}.pt"


def _atomic_write_new(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` only if ``path`` does not exist yet.

    Writes to a temp file in the same directory, then hard-links it into place.
    ``os.link`` fails if the target exists, so an existing checkpoint can never
    be clobbered, and readers never see a half-written file.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def save_new_checkpoint(bundle: ModelBundle, checkpoint_dir: str | os.PathLike, name: str | None = None) -> tuple[str, Path]:
    """Persist ``bundle`` as a new file. Returns (version, path). Refuses to overwrite."""
    d = Path(checkpoint_dir)
    d.mkdir(parents=True, exist_ok=True)
    version = bundle.version()
    bundle.meta["version"] = version
    path = d / (name or ckpt_name(version))
    if path.exists():
        raise CheckpointError(f"{path} already exists; checkpoints are never overwritten")
    _atomic_write_new(path, bundle.to_bytes())
    return version, path


def ensure_init_checkpoint(
    checkpoint_dir: str | os.PathLike, config: ModelConfig | None = None, seed: int = INIT_SEED
) -> Path:
    """Return the path of ``init.pt``, creating it (seeded random init) if missing."""
    d = Path(checkpoint_dir)
    path = d / INIT_NAME
    if path.exists():
        return path
    bundle = ModelBundle.create(config or ModelConfig.default(), seed)
    try:
        save_new_checkpoint(bundle, d, name=INIT_NAME)
    except FileExistsError:
        pass  # another process created it between our check and our link; theirs wins
    return path


def serving_checkpoint_path(checkpoint_dir: str | os.PathLike) -> Path:
    """The checkpoint users currently see: ``current`` if set, else ``init.pt``."""
    d = Path(checkpoint_dir)
    pointer = d / CURRENT_NAME
    if pointer.exists():
        name = pointer.read_text().strip()
        path = d / name
        if not name or "/" in name or not path.exists():
            raise CheckpointError(f"{pointer} points at {name!r}, which does not exist in {d}")
        return path
    return d / INIT_NAME


def checkpoint_path_for(checkpoint_dir: str | os.PathLike, version: str) -> Path:
    """Find the file for ``version`` (a ckpt-<version>.pt, or init.pt if that is its version)."""
    d = Path(checkpoint_dir)
    path = d / ckpt_name(version)
    if path.exists():
        return path
    init = d / INIT_NAME
    if init.exists() and read_meta(init).get("version") == version:
        return init
    raise CheckpointError(f"no checkpoint with version {version!r} in {d}")


def read_meta(path: str | os.PathLike) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=True)
    return dict(data.get("meta") or {})


def list_checkpoints(checkpoint_dir: str | os.PathLike) -> list[dict]:
    """Every checkpoint in the directory with its meta, oldest first, marking the served one."""
    d = Path(checkpoint_dir)
    if not d.exists():
        return []
    try:
        serving = serving_checkpoint_path(d).name
    except CheckpointError:
        serving = None
    out = []
    for path in sorted(d.glob("*.pt"), key=lambda p: p.stat().st_mtime):
        meta = read_meta(path)
        out.append({"file": path.name, "version": meta.get("version"), "serving": path.name == serving, **meta})
    return out


def promote(checkpoint_dir: str | os.PathLike, version: str) -> Path:
    """Make ``version`` the served checkpoint by atomically rewriting ``current``."""
    d = Path(checkpoint_dir)
    path = checkpoint_path_for(d, version)
    ModelBundle.from_file(path)  # refuse to promote a file that does not load
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".current.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(path.name + "\n")
    os.replace(tmp, d / CURRENT_NAME)
    return path


class GlyphEngine:
    """Turns intents into glyphs with the served checkpoint."""

    def __init__(self, bundle: ModelBundle, path: Path, device: torch.device | str = "cpu"):
        self.bundle = bundle.to(device)
        self.bundle.speaker.eval()
        self.path = Path(path)
        self.device = torch.device(device)
        self.model_version: str = bundle.meta.get("version") or bundle.version()

    @classmethod
    def load(
        cls,
        checkpoint_dir: str | os.PathLike,
        device: torch.device | str = "cpu",
        config: ModelConfig | None = None,
    ) -> "GlyphEngine":
        """Load whatever is being served; on an empty directory create and persist ``init.pt``.

        ``config`` only matters when ``init.pt`` has to be created (tests use
        ``ModelConfig.tiny()``); an existing checkpoint always carries its own.
        """
        d = Path(checkpoint_dir)
        d.mkdir(parents=True, exist_ok=True)
        if not (d / CURRENT_NAME).exists():
            ensure_init_checkpoint(d, config)
        path = serving_checkpoint_path(d)
        return cls(ModelBundle.from_file(path), path, device)

    @property
    def n_concepts(self) -> int:
        return self.bundle.config.n_concepts

    def speak(self, intent: Intent, variation: int | None = None) -> Glyph:
        """Draw ``intent``. Same intent + same model_version (+ same variation) -> same glyph.

        ``variation`` seeds a small perturbation of the speaker's stroke slots,
        giving a different-but-related drawing of the same meaning (useful so a
        repeated intent does not always look pixel-identical). ``None`` means
        the canonical glyph.
        """
        for name in intent.concepts:
            if concepts.by_name(name).id >= self.n_concepts:
                raise CheckpointError(
                    f"atom {name!r} was added to the vocabulary after checkpoint {self.model_version} "
                    "was trained; train a new checkpoint before using it"
                )
        cfg = self.bundle.config
        ids, mask = pack_intents([intent], self.device)
        noise = None
        if variation is not None:
            gen = torch.Generator().manual_seed(int(variation))
            noise = (torch.randn(1, cfg.n_strokes, cfg.d_model, generator=gen) * cfg.variation_scale).to(self.device)
        with torch.no_grad():
            strokes = self.bundle.speaker(ids, mask, noise)[0].to("cpu", torch.float64)
        rows = [[round(float(v), _DECIMALS) for v in row] for row in strokes.tolist()]
        return Glyph(
            strokes=tuple(Stroke.from_list(r) for r in rows),
            model_version=self.model_version,
            variation=variation,
        )
