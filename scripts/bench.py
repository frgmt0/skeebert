"""Benchmarks for the public site: every number on skeebert.frgmt.xyz's benchmark
section comes out of this script (or out of the training log it summarises).

It loads two checkpoints READ-ONLY and runs inference only:

* the untrained baseline ``checkpoints/init.pt`` (seeded random init), and
* the trained bootstrap checkpoint ``checkpoints/ckpt-6d99ccb9bda1.pt``.

Nothing is trained and nothing is written under ``checkpoints/``. Evaluation
reuses ``skeebert.train``'s own helpers (``sample_intents``,
``make_distractors``, ``game_loss``, ``augment``, ``topographic_similarity``
and the stage's ``_eval_game``), so the numbers mean exactly what the logged
training metrics mean.

Writes:

* ``site/public/bench.json``    every measured number, with seeds, n, device,
                                 date, checkpoint versions and the git commit;
* ``site/public/trainlog.json`` a summary of the bootstrap run's JSONL log
                                 (loss/accuracy curve, listener resets, final eval).

Usage::

    uv run python scripts/bench.py [--checkpoint-dir checkpoints] [--out site/public]
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import torch

from skeebert import concepts
from skeebert.glyph import INIT_NAME, GlyphEngine
from skeebert.model import ModelBundle, multi_hot
from skeebert.render import render
from skeebert.train import (
    TrainConfig,
    _Stage,
    augment,
    game_loss,
    make_distractors,
    sample_intents,
    topographic_similarity,
)
from skeebert.types import MAX_INTENT_ATOMS, Intent

TRAINED_FILE = "ckpt-6d99ccb9bda1.pt"

# Fixed seeds. Changing any of these changes the published numbers.
SEED_INTENTS = 1234  # intents for the candidate-count sweep
SEED_DISTRACTORS = 4321  # distractor sets (shared by both checkpoints)
SEED_AUG = 777  # augmentation noise for the "augmented" sweep
SEED_TOPSIM = (0, 1, 2, 3, 4)
SEED_DISTINCT = 99
SEED_GLYPHBENCH = 7
SEED_LATENCY = 2026

N_SWEEP = 2048
CANDIDATE_COUNTS = (2, 4, 8, 16, 32)
N_TOPSIM = 512
N_DISTINCT_INTENTS = 128
N_VARIATIONS = 8
N_GLYPHBENCH = 1000
N_LATENCY_WARMUP = 20
N_LATENCY = 200
N_RENDER_LATENCY = 50
BATCH = 64

GLYPHBENCH_RULE = (
    "Eligibility: the system's entire answer must be a single glyph of at most 6 quadratic strokes, "
    "with no words. Eligible systems draw 1,000 intents; skeebert's own trained listener then picks "
    "the intended meaning out of 16 candidates (half of the distractors differ by one concept) from the "
    "glyph alone, after the training-time augmentation (rotation, scale, shift, pixel noise). "
    "Score = share picked correctly. Ineligible systems score 0% by rule; they are not run."
)

# Facts that are not computed here because they live elsewhere (the production
# host's live database, the operator's API ledger). Reported verbatim, with source.
REPORTED = {
    "as_of": "2026-10-08",
    "brain": {
        "model": "Claude Haiku 5.5",
        "model_id": "claude-haiku-5-5",
        "method": "structured output over the 218-concept enum",
        "source": "skeebert/brain/haiku.py",
    },
    "ledger": {
        "calls": 4,
        "usd": 0.0008,
        "tokens_in": 408,
        "tokens_out": 102,
        "cache_read": 14505,
        "cache_write": 4835,
        "source": "live spend ledger on the production host",
    },
    "persona_tuning": {"calls": 55, "usd": 0.0062, "source": "operator's tuning session, 2026-10-08"},
    "daily_budget_usd": 6.0,
    "human_eval": {
        "guesses": 4,
        "people": 1,
        "mean_score": 0.346,
        "unlock_threshold": 300,
        "score_definition": "rescaled MiniLM cosine with per-concept partial credit (skeebert/scoring.py)",
        "source": "live database on the production host",
    },
    "prices_per_mtok_usd": {
        "input": 0.10,
        "output": 0.50,
        "cache_read": 0.01,
        "cache_write": 0.125,
        "source": "README.md, Budget and sleep (configured defaults)",
    },
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def r(x: float, nd: int = 4) -> float:
    return float(round(float(x), nd))


def prop(k: float, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"value": r(k / n), "k": int(round(k)), "n": n, "ci95": [r(lo), r(hi)]}


def git_commit(root: Path) -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {"commit": run("rev-parse", "HEAD") or None, "short": run("rev-parse", "--short", "HEAD") or None,
            "dirty": bool(run("status", "--porcelain"))}


def images_for(bundle: ModelBundle, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    cfg = bundle.config
    out = []
    with torch.no_grad():
        for s in range(0, ids.shape[0], BATCH):
            strokes = bundle.speaker(ids[s : s + BATCH], mask[s : s + BATCH])
            out.append(render(strokes, cfg.image_size, segments=cfg.render_segments))
    return torch.cat(out)


def logits_for(bundle: ModelBundle, images: torch.Tensor) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for s in range(0, images.shape[0], BATCH):
            out.append(bundle.listener(images[s : s + BATCH]))
    return torch.cat(out)


def candidate_hits(logits: torch.Tensor, target: torch.Tensor, cands: torch.Tensor) -> float:
    """Number of items where the target outranks every candidate (train.game_loss's accuracy x n)."""
    hits = 0.0
    for s in range(0, logits.shape[0], BATCH):
        _, _, acc = game_loss(logits[s : s + BATCH], target[s : s + BATCH], cands[s : s + BATCH])
        hits += float(acc) * logits[s : s + BATCH].shape[0]
    return hits


# ---------------------------------------------------------------------------
# training log summary
# ---------------------------------------------------------------------------

def summarise_log(log_path: Path) -> dict:
    start, eta, done, points = None, None, None, []
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ev = rec.get("event")
        if ev == "start":
            start = rec
        elif ev == "eta":
            eta = rec
        elif ev == "done":
            done = rec
        elif ev == "step":
            points.append([rec["step"], r(rec["elapsed"], 1), r(rec["loss"]), r(rec["game_ce"]), r(rec["bce"]),
                           r(rec["acc"]), r(rec["ink"], 5)])
    if start is None or done is None:
        raise SystemExit(f"{log_path}: incomplete log (no start or done event)")
    cfg = start["config"]
    every, steps = cfg["listener_reset_every"], cfg["steps"]
    # Same rule as train._Bootstrap._maybe_reset_listener: no reset in the final interval.
    resets = [s for s in range(every, steps, every) if s + every <= steps] if every > 0 else []

    def window_acc(lo: int, hi: int) -> float | None:
        vals = [p[5] for p in points if lo < p[0] <= hi]
        return r(statistics.fmean(vals)) if vals else None

    def steps_to(level: float, lo: int, hi: int, window: int = 10) -> int | None:
        """Steps after ``lo`` until the rolling mean of logged batch accuracy first reaches ``level``."""
        seg = [p for p in points if lo < p[0] <= hi]
        for i in range(window - 1, len(seg)):
            if statistics.fmean(q[5] for q in seg[i - window + 1 : i + 1]) >= level:
                return seg[i][0] - lo
        return None

    generations = []
    bounds = [0, *resets, steps]
    for g, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        generations.append({
            "generation": g + 1, "from_step": lo, "to_step": hi,
            "acc_first_500": window_acc(lo, lo + 500),
            "acc_last_500": window_acc(hi - 500, hi),
            "steps_to_rolling_0_5": steps_to(0.5, lo, hi),
        })
    return {
        "source": f"checkpoints/logs/{log_path.name}",
        "stage": start["stage"],
        "parent": start["parent"],
        "version": done["version"],
        "device": start["device"],
        "steps": steps,
        "config": cfg,
        "params": start["params"],
        "eta_probe": {"seconds_per_step": r(eta["seconds_per_step"]), "eta": eta["eta"]} if eta else None,
        "seconds": r(done["seconds"], 1),
        "seconds_per_step": r(done["seconds"] / steps),
        "final_eval": {k: r(v) for k, v in done["metrics"].items()},
        "listener_reset_steps": resets,
        "generations": generations,
        "columns": ["step", "elapsed_s", "loss", "game_ce", "bce", "acc", "ink"],
        "points": points,
        "batch_size": cfg["batch_size"],
        "rolling_window_points": 10,
        "note": "acc is the per-batch training accuracy (augmented images, 16 candidates) logged every "
                f"{cfg['log_every']} steps; one batch = {cfg['batch_size']} intents.",
    }


# ---------------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------------

def bench_sweep(models: dict[str, ModelBundle], n_concepts: int, probs) -> dict:
    gen = torch.Generator().manual_seed(SEED_INTENTS)
    ids, mask = sample_intents(N_SWEEP, n_concepts, probs, gen)
    target = multi_hot(ids, mask, n_concepts)
    cfg = TrainConfig()
    clean, aug = {}, {}
    for name, b in models.items():
        imgs = images_for(b, ids, mask)
        clean[name] = logits_for(b, imgs)
        aug[name] = logits_for(b, augment(imgs, cfg, torch.Generator().manual_seed(SEED_AUG)))

    rows = []
    for k in CANDIDATE_COUNTS:
        dgen = torch.Generator().manual_seed(SEED_DISTRACTORS + k)
        cands = make_distractors(ids, mask, k - 1, n_concepts, cfg.hard_distractor_frac, probs, dgen)
        row = {"candidates": k, "chance": r(1.0 / k)}
        for name in models:
            row[name] = prop(candidate_hits(clean[name], target, cands), N_SWEEP)
            row[name + "_augmented"] = prop(candidate_hits(aug[name], target, cands), N_SWEEP)
        rows.append(row)

    # distractor type, at 16 candidates
    by_type = []
    for label, frac in (("one-concept-off only", 1.0), ("random only", 0.0), ("mixed 50/50 (training mix)", 0.5)):
        dgen = torch.Generator().manual_seed(SEED_DISTRACTORS + 1000 + int(frac * 10))
        cands = make_distractors(ids, mask, 15, n_concepts, frac, probs, dgen)
        row = {"distractors": label, "hard_frac": frac}
        for name in models:
            row[name] = prop(candidate_hits(clean[name], target, cands), N_SWEEP)
        by_type.append(row)

    # exact-set accuracy (every one of the 218 atom decisions right) by intent size
    sizes = mask.sum(1)
    exact = []
    for size in range(1, MAX_INTENT_ATOMS + 1):
        sel = sizes == size
        n = int(sel.sum())
        row = {"atoms": size, "n": n}
        for name in models:
            ok = ((clean[name][sel] > 0).float() == target[sel]).all(-1).float().sum()
            row[name] = prop(float(ok), n)
        exact.append(row)
    allrow = {"atoms": "all", "n": N_SWEEP}
    for name in models:
        allrow[name] = prop(float(((clean[name] > 0).float() == target).all(-1).float().sum()), N_SWEEP)
    exact.append(allrow)

    return {
        "n": N_SWEEP, "seed_intents": SEED_INTENTS, "seed_distractors": SEED_DISTRACTORS, "seed_aug": SEED_AUG,
        "atom_count_probs": list(probs), "hard_distractor_frac": cfg.hard_distractor_frac,
        "candidate_sweep": rows, "distractor_type_16": by_type, "exact_set_by_size": exact,
    }


def bench_atoms(models: dict[str, ModelBundle], n_concepts: int) -> dict:
    """Single-concept glyphs: can the listener name the concept, top-1 of all 218?"""
    ids = torch.zeros(n_concepts, MAX_INTENT_ATOMS, dtype=torch.long)
    ids[:, 0] = torch.arange(n_concepts)
    mask = torch.zeros(n_concepts, MAX_INTENT_ATOMS, dtype=torch.bool)
    mask[:, 0] = True
    cats = [concepts.by_id(i).category for i in range(n_concepts)]
    out = {"n": n_concepts, "chance_top1": r(1 / n_concepts, 5), "chance_top5": r(5 / n_concepts, 5), "categories": []}
    ranks = {}
    for name, b in models.items():
        lg = logits_for(b, images_for(b, ids, mask))
        true = lg.gather(1, torch.arange(n_concepts).unsqueeze(1))
        ranks[name] = (lg > true).sum(1)  # 0 = top-1
        out[name] = {"top1": prop(float((ranks[name] == 0).sum()), n_concepts),
                     "top5": prop(float((ranks[name] < 5).sum()), n_concepts)}
    for cat in concepts.CATEGORIES:
        sel = torch.tensor([c == cat for c in cats])
        n = int(sel.sum())
        if not n:
            continue
        row = {"category": cat, "n": n}
        for name in models:
            row[name] = prop(float((ranks[name][sel] == 0).sum()), n)
        out["categories"].append(row)
    return out


def bench_topsim(models: dict[str, ModelBundle], n_concepts: int, probs) -> dict:
    out = {"n_per_seed": N_TOPSIM, "seeds": list(SEED_TOPSIM),
           "definition": "Spearman correlation between Jaccard meaning distance and L2 glyph distance over all "
                         "pairs (skeebert.train.topographic_similarity), 64px renders"}
    for name, b in models.items():
        vals = []
        for seed in SEED_TOPSIM:
            ids, mask = sample_intents(N_TOPSIM, n_concepts, probs, torch.Generator().manual_seed(seed))
            vals.append(topographic_similarity(multi_hot(ids, mask, n_concepts), images_for(b, ids, mask)))
        out[name] = {"mean": r(statistics.fmean(vals)), "sd": r(statistics.stdev(vals)), "per_seed": [r(v) for v in vals]}
    return out


def bench_logged_eval(models: dict[str, ModelBundle], device: str) -> dict:
    """Re-run the exact end-of-run eval from train.py (seed 10000, 512 intents, 16 candidates)."""
    out = {"procedure": "skeebert.train._Stage._eval_game with TrainConfig() defaults", "device": device}
    for name, b in models.items():
        stage = _Stage(b, TrainConfig(), torch.device(device))
        out[name] = {k: r(v) for k, v in stage._eval_game().items()}
    return out


def bench_distinct(engines: dict[str, GlyphEngine], n_concepts: int, probs) -> dict:
    """Mean per-pixel |difference| between glyphs: different meanings vs re-draws of the same meaning."""
    ids, mask = sample_intents(N_DISTINCT_INTENTS, n_concepts, probs, torch.Generator().manual_seed(SEED_DISTINCT))
    intents = [Intent.from_ids(int(i) for i, m in zip(row.tolist(), mrow.tolist()) if m) for row, mrow in zip(ids, mask)]
    intents = list(dict.fromkeys(intents))
    out = {"intents": len(intents), "variations_per_intent": N_VARIATIONS, "size_px": 64, "seed": SEED_DISTINCT,
           "metric": "mean absolute per-pixel difference in ink coverage (0 = identical, 1 = opposite)",
           "variation": "GlyphEngine.speak(intent, variation=v), v = 0..7 (speaker slot noise, scale 0.35)"}
    for name, eng in engines.items():
        canon = torch.stack([eng.speak(it).coverage(64) for it in intents]).float()
        flat = canon.reshape(len(intents), -1)
        between = torch.cdist(flat, flat, p=1) / flat.shape[1]
        iu = torch.triu_indices(len(intents), len(intents), 1)
        between_vals = between[iu[0], iu[1]]
        within_vals = []
        for it in intents:
            vs = torch.stack([eng.speak(it, variation=v).coverage(64) for v in range(N_VARIATIONS)]).float()
            vf = vs.reshape(N_VARIATIONS, -1)
            for a, c in combinations(range(N_VARIATIONS), 2):
                within_vals.append(float((vf[a] - vf[c]).abs().mean()))
        b_mean = float(between_vals.mean())
        w_mean = statistics.fmean(within_vals)
        out[name] = {"between_meanings": r(b_mean, 5), "between_pairs": int(between_vals.numel()),
                     "within_meaning": r(w_mean, 5), "within_pairs": len(within_vals),
                     "ratio": r(b_mean / w_mean, 3) if w_mean > 0 else None}
    return out


def bench_glyphbench(models: dict[str, ModelBundle], n_concepts: int, probs) -> dict:
    gen = torch.Generator().manual_seed(SEED_GLYPHBENCH)
    ids, mask = sample_intents(N_GLYPHBENCH, n_concepts, probs, gen)
    target = multi_hot(ids, mask, n_concepts)
    cfg = TrainConfig()
    cands = make_distractors(ids, mask, 15, n_concepts, 0.5, probs, torch.Generator().manual_seed(SEED_GLYPHBENCH + 1))
    reader = models["trained"]  # the fixed reader for every entrant
    out = {"name": "GlyphBench v1", "rule": GLYPHBENCH_RULE, "n": N_GLYPHBENCH, "candidates": 16, "chance": r(1 / 16),
           "seed": SEED_GLYPHBENCH, "reader": "listener of checkpoint " + reader.meta.get("version", "?"),
           "entrants": []}
    for name, b in models.items():
        imgs = images_for(b, ids, mask)
        lg = logits_for(reader, augment(imgs, cfg, torch.Generator().manual_seed(SEED_GLYPHBENCH + 2)))
        out["entrants"].append({"name": name, "version": b.meta.get("version"), "eligible": True,
                                "score": prop(candidate_hits(lg, target, cands), N_GLYPHBENCH)})
    for frontier in ("Opus 5.5", "GPT 6.1 Sol", "Gemini 4 Argon"):
        out["entrants"].append({"name": frontier, "eligible": False, "score": None, "score_by_rule": 0.0,
                                "note": "answers in words; not run"})
    return out


def bench_latency(engine: GlyphEngine, n_concepts: int, probs) -> dict:
    ids, mask = sample_intents(N_LATENCY_WARMUP + N_LATENCY, n_concepts, probs,
                               torch.Generator().manual_seed(SEED_LATENCY))
    intents = [Intent.from_ids(int(i) for i, m in zip(row.tolist(), mrow.tolist()) if m) for row, mrow in zip(ids, mask)]
    speak_ms, glyphs = [], []
    for i, it in enumerate(intents):
        t = time.perf_counter()
        g = engine.speak(it)
        dt = (time.perf_counter() - t) * 1000
        if i >= N_LATENCY_WARMUP:
            speak_ms.append(dt)
            glyphs.append(g)
    png_ms, raw_ms, png_bytes = [], [], []
    for g in glyphs[:N_RENDER_LATENCY]:
        t = time.perf_counter()
        data = g.png_bytes(512)
        png_ms.append((time.perf_counter() - t) * 1000)
        png_bytes.append(len(data))
        tensor = g.to_tensor()
        t = time.perf_counter()
        with torch.no_grad():
            render(tensor, 512)
        raw_ms.append((time.perf_counter() - t) * 1000)

    def stats(xs):
        s = sorted(xs)
        return {"n": len(xs), "median_ms": r(statistics.median(s), 2), "p95_ms": r(s[int(0.95 * (len(s) - 1))], 2),
                "mean_ms": r(statistics.fmean(s), 2)}

    return {"device": "cpu", "torch_threads": torch.get_num_threads(), "warmup": N_LATENCY_WARMUP,
            "speak": stats(speak_ms), "render_512_tensor": stats(raw_ms), "png_512_export": stats(png_ms),
            "png_bytes_median": int(statistics.median(png_bytes)),
            "notes": "speak = GlyphEngine.speak (speaker forward, one intent, no batching); "
                     "render_512_tensor = render(strokes, 512); png_512_export = Glyph.png_bytes(512), "
                     "the exact image sent to Discord (render + colorize + PNG encode)"}


SCORING_INTENT = ("food", "question")
SCORING_GUESSES = ("are you hungry?", "food", "do you want something to eat?", "a question", "hello", "I love dogs")


def bench_scoring() -> dict:
    """score_guess with the real embedder (all-MiniLM-L6-v2) on fixed guesses. Offline if cached."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from skeebert.embed import Embedder
    from skeebert.scoring import CEIL, FLOOR, WHOLE_WEIGHT, score_guess

    try:
        emb = Embedder()
        rows = [{"guess": g, "score": r(score_guess(Intent(SCORING_INTENT), g, emb), 3)} for g in SCORING_GUESSES]
    except Exception as exc:  # no cached model and no network: say so instead of inventing numbers
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    return {"available": True, "intent": list(SCORING_INTENT), "gloss": Intent(SCORING_INTENT).gloss(),
            "embedder": emb.model_name, "floor": FLOOR, "ceil": CEIL, "whole_weight": WHOLE_WEIGHT, "rows": rows}


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--out", type=Path, default=Path("site/public"))
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    ck: Path = args.checkpoint_dir
    paths = {"untrained": ck / INIT_NAME, "trained": ck / TRAINED_FILE}
    for p in paths.values():
        if not p.exists():
            print(f"error: {p} not found", file=sys.stderr)
            return 2
    logs = sorted((ck / "logs").glob("*-bootstrap-*.jsonl"))
    if not logs:
        print(f"error: no bootstrap log in {ck / 'logs'}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    torch.manual_seed(0)
    device = "cpu"  # deterministic, and what the bot serves on
    models = {name: ModelBundle.from_file(p, device) for name, p in paths.items()}
    for b in models.values():
        b.speaker.eval(), b.listener.eval(), b.proxy.eval()
    versions = {name: (b.meta.get("version") or b.version()) for name, b in models.items()}
    n_concepts = models["trained"].config.n_concepts
    probs = TrainConfig().atom_count_probs
    timings = {}

    def timed(label, fn, *a):
        t = time.perf_counter()
        res = fn(*a)
        timings[label] = r(time.perf_counter() - t, 1)
        print(f"  {label}: {timings[label]}s", flush=True)
        return res

    print("benchmarking", ", ".join(f"{k}={v}" for k, v in versions.items()), flush=True)
    sweep = timed("sweep", bench_sweep, models, n_concepts, probs)
    atoms = timed("atoms", bench_atoms, models, n_concepts)
    topsim = timed("topsim", bench_topsim, models, n_concepts, probs)
    logged = timed("logged_eval", bench_logged_eval, models, device)
    glyphbench = timed("glyphbench", bench_glyphbench, models, n_concepts, probs)
    engines = {name: GlyphEngine(b, paths[name], device) for name, b in models.items()}
    distinct = timed("distinct", bench_distinct, engines, n_concepts, probs)
    latency = timed("latency", bench_latency, engines["trained"], n_concepts, probs)
    scoring = timed("scoring", bench_scoring)
    trainlog = summarise_log(logs[-1])

    cfg = models["trained"].config
    counts = models["trained"].param_counts()
    out = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "scripts/bench.py",
        "git": git_commit(root),
        "device": device,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
        "checkpoints": {name: {"file": paths[name].name, "version": versions[name],
                               "stage": models[name].meta.get("stage")} for name in models},
        "model": {
            "params": counts,
            "n_strokes": cfg.n_strokes,
            "params_per_stroke": 8,
            "image_size_train": cfg.image_size,
            "export_size": 512,
            "n_concepts": n_concepts,
            "n_categories": len(concepts.CATEGORIES),
            "categories": [{"name": c, "n": sum(1 for x in concepts.CONCEPTS[:n_concepts] if x.category == c)}
                           for c in concepts.CATEGORIES],
            "max_atoms": MAX_INTENT_ATOMS,
            "possible_messages": sum(math.comb(n_concepts, k) for k in range(1, MAX_INTENT_ATOMS + 1)),
            "d_model": cfg.d_model, "enc_layers": cfg.enc_layers, "dec_layers": cfg.dec_layers,
            "n_heads": cfg.n_heads, "channels": list(cfg.channels), "hidden": cfg.hidden,
            "render_segments": cfg.render_segments,
        },
        "logged_eval_rerun": logged,
        "sweep": sweep,
        "atoms": atoms,
        "topsim": topsim,
        "glyphbench": glyphbench,
        "distinctness": distinct,
        "latency": latency,
        "scoring": scoring,
        "training_run": {k: trainlog[k] for k in ("source", "version", "parent", "device", "steps", "seconds",
                                                    "seconds_per_step", "final_eval", "listener_reset_steps",
                                                    "batch_size")},
        "reported": REPORTED,
        "timings_s": timings,
    }
    out["runtime_s"] = r(time.perf_counter() - t0, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bench.json").write_text(json.dumps(out, indent=1) + "\n")
    (args.out / "trainlog.json").write_text(json.dumps(trainlog, separators=(",", ":")) + "\n")
    print(f"wrote {args.out / 'bench.json'} and {args.out / 'trainlog.json'} in {out['runtime_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
