# Skeebert's ML core

Skeebert has two halves. The **brain** (an LLM, built elsewhere) decides *what*
to say as an intent: one to three atoms from a fixed vocabulary. The **mouth**
is a small model trained from scratch here. It turns the intent into a glyph,
a handful of strokes rendered to an image. The LLM never sees or picks glyph
shapes. Whatever visual language exists is one that Skeebert and the humans
guessing in Discord built between them.

This document covers how that language is supposed to emerge, how training
works, the checkpoint rules, and what is not done yet.

## Pieces

| module | what it is |
| --- | --- |
| `skeebert/concepts.py` | 218 atoms in 9 categories (people, feeling, speech, action, thing, place, time, quantity, modifier). **Append-only**: an atom's id is its position, so never reorder, rename or delete. Only add to the end. A test pins a fingerprint of the existing names. |
| `skeebert/types.py` | `Intent` (1–3 atoms, kept in sorted order), `Stroke`, `Glyph` (strokes + `model_version`, JSON, PNG/JPEG), `TrainingExample`. |
| `skeebert/render.py` | Differentiable rasterizer. K quadratic Bézier strokes become a soft distance field. Training renders at 64px and export at 512px, using the same function. |
| `skeebert/model.py` | `Speaker`, `Listener`, `HumanProxy`, `ModelConfig`, and `ModelBundle` (one checkpoint's worth of all three). |
| `skeebert/glyph.py` | `GlyphEngine` (serving) plus the checkpoint-directory rules: init, save, promote, list. |
| `skeebert/embed.py` | `Embedder` (all-MiniLM-L6-v2, 384-d, loaded lazily) and the `TextEmbedder` protocol. |
| `skeebert/scoring.py` | `score_guess`: guess vs. intent, in [0, 1], with partial credit per atom. |
| `skeebert/train.py` | The three training stages, runtime estimation, and the confirm-gated CLI entry. |

## The glyph

A glyph is K strokes (6 by default). Each stroke has 8 numbers: three Bézier
control points, a half-width and an intensity. A stroke whose intensity is
near 0 lays down no ink, so the model can draw fewer than K strokes. The
renderer works out each pixel's distance to each stroke and passes
`half_width − distance` through a sigmoid whose softness is set in *pixels*.
The result is that 64px and 512px renders are the same continuous picture,
each anti-aliased at its own resolution. A test checks that a 512 render
box-downsampled to 64 matches the 64 render. This matters because humans must
judge the same picture the model was trained on.

The model only ever sees single-channel ink coverage. Colour is applied at
export time (`render.colorize`): pale bone-white strokes with a soft amber
glow on a blue-black field. It is cosmetic only.

## How the language is meant to emerge

### Why compositionality is plausible here

- **Set-structured speaker.** Atom embeddings go through a transformer
  encoder with no positional encoding, so `food+question` and
  `question+food` give the same glyph. K learned stroke-slot queries then
  cross-attend to the atoms, one stroke per slot. A slot can learn to follow
  one atom, which means the glyph for food+question can be the food strokes
  plus the question strokes. Each slot also has its own learned default
  stroke (`slot_bias`), so untrained strokes start out spread around the
  canvas instead of piled on top of each other.
- **Limited strokes.** With 6 strokes for up to 3 atoms there is no room to
  give each of the roughly 1.6M possible intents its own unrelated picture.
  Reusing parts is the cheapest way to fit.
- **Hard distractors.** In the game, half the distractors are the target with
  one atom swapped, dropped or added. Telling `food+question` apart from
  `food+yes` is only possible if the question part can be seen on its own.
- **Augmentation.** Every image a listener or proxy sees during training gets
  a small random rotation, scale, shift and pixel noise. A language that only
  works pixel-perfectly would fail on a phone screen through Discord's JPEG
  compression, so augmentation pushes toward bold, shape-level differences.
- **Iterated learning.** Every `listener_reset_every` steps (default 5000)
  the listener is swapped for a freshly initialised one. Languages built from
  reusable parts are faster for a new learner to pick up, so over many
  generations the speaker drifts toward systematic glyphs (Ren et al. 2020).
  No reset happens in the last interval, so the saved listener can actually
  read the saved speaker.
- **Ink cost.** A small penalty on mean ink keeps glyphs sparse enough to
  read.

Compositionality is measured by **topographic similarity** (`eval_topsim`):
the Spearman correlation between meaning distance (Jaccard over atom sets)
and glyph distance (L2 between images). It sits near 0 for random glyphs and
rises as intents that share atoms start sharing strokes.

### Stage 1: `bootstrap` (self-play, no humans)

The speaker draws random intents (40% one atom, 35% two, 25% three). The
listener outputs one logit per atom. Each candidate intent (the target plus
15 distractors) is scored by its log-likelihood under those independent
per-atom Bernoulli outputs. The loss is cross-entropy over the candidates,
plus multi-label BCE on the atoms, plus the ink penalty. Speaker and listener
train together. If a distractor happens to equal the target, it is masked
out.

The point of this stage is to give Skeebert a *consistent* language before
anyone sees it, so the same meaning comes out as recognisably the same strokes
and humans have something learnable to latch onto. Without it, the first few
hundred human guesses would be scored against glyphs that are effectively
random per intent.

### Stage 2: `proxy` (fit a model of human readers)

`HumanProxy` maps the image of the **exact glyph that was shown**,
re-rendered from its stored strokes at 64px, to the sentence embedding of
what the human typed. It is trained on every guess, right or wrong, because
it models what people *see*, not what Skeebert meant. Blank guesses are
skipped. Ten percent is held out. The report includes `val_cos` next to
`val_baseline_cos`, which is what you get by predicting the average guess for
every glyph. A proxy that does not beat the baseline has only learned "people
usually say X", and aligning to it would be pointless. Check this before
running stage 3.

Only the proxy's weights change in this stage. The speaker is untouched, so
glyphs look the same, but the checkpoint still gets a new version because its
weights differ.

### Stage 3: `align` (pull the speaker toward human readings)

The speaker is fine-tuned to minimise `1 − cos(HumanProxy(render(Speaker(intent))), embed(intent.gloss()))`,
which means drawing glyphs that the proxy predicts humans would read
correctly. Proxy and listener are frozen. The listener's game loss (weight
`align_game_weight`, default 0.5) acts as a regulariser: it keeps the glyphs
readable to the established reader, so the language cannot collapse into
whatever happens to fool the proxy. Half of each batch comes from intents
that appear in the human data and half from random intents, so the alignment
does not forget the rest of the vocabulary.

### Data loop

The bot shows a glyph. A human guesses. `score_guess` compares the guess to
the intent and the result is stored as a `TrainingExample` (intent, the exact
glyph JSON, guess text, score). Periodically, a human runs proxy and then
align on the accumulated examples, reviews the metrics and promotes.

## Scoring

```
whole    = cos(guess, intent.gloss())
coverage = mean over atoms of cos(guess, atom.gloss)
raw      = 0.5·whole + 0.5·coverage
score    = clip((raw − 0.15) / (0.80 − 0.15), 0, 1)
```

`coverage` gives partial credit: guessing "food" for food+question scores
about half. The rescale stretches MiniLM's useful cosine band onto [0, 1].
FLOOR and CEIL are calibration constants based on how the model generally
behaves. They have not been fitted to Skeebert data, because none exists yet.
A one-off manual check with the real model for food+question gave: "food"
0.83, "do you want something to eat?" 0.56, "a question" 0.44, "hello" 0.13,
"I love dogs" 0.01.

## Checkpoints and versioning

```
checkpoints/
  init.pt              seeded random init (seed 1729), created once, never rewritten
  ckpt-<version>.pt    every trained model, one file per version, never overwritten
  current              names the served file; if absent, init.pt is served
  logs/<utc>-<stage>-<parent>.jsonl   per-run loss log
```

- `model_version` is the first 12 hex characters of a sha256 over the config
  and all weights. Every glyph records it, and it always names a file that
  exists.
- `GlyphEngine.load(dir)` on an empty directory creates and **persists**
  `init.pt`. The untrained Skeebert's glyphs therefore stay the same across
  restarts, and the earliest stored glyphs point at a real checkpoint.
- Training starts from the served checkpoint, or from `base=<version>`.
  It always writes a new `ckpt-<version>.pt` through a temp file and a hard
  link, so an existing file can never be clobbered. It **never** touches
  `current`.
- `glyph.promote(dir, version)` is the only thing that changes what is
  served. It checks that the file loads, then atomically rewrites `current`.
  To roll back to the untrained model, promote init's version.
- Atoms added later (append-only) are handled like this: training grows the
  speaker's embedding table and the listener's output layer, keeping the
  existing rows. Serving an intent that uses an atom newer than the
  checkpoint raises a clear error.
- A stage that hits a non-finite loss raises `TrainingDiverged` and saves
  nothing.
- The version hashes weights and `ModelConfig`, but not the drawing-surface
  constants in `render.py` (`COORD_LIMIT`, `MIN/MAX_HALF_WIDTH`) or the
  colours. Changing those changes how every glyph looks without changing any
  version. Treat them as frozen once real glyphs have been shown to people.

## Running training (API for the CLI)

```python
from skeebert.train import TrainConfig, estimate_runtime, train_from_cli

est = estimate_runtime(examples, "bootstrap", TrainConfig.for_stage("bootstrap"), "checkpoints")
print(est.describe())                       # times a few real steps; writes nothing
report = train_from_cli(examples, "bootstrap", confirm=True, checkpoint_dir="checkpoints")
# then, only after looking at report.metrics:
from skeebert.glyph import promote; promote("checkpoints", report.version)
```

`train_from_cli` raises `TrainingNotConfirmed` unless `confirm=True`. Proxy
and align raise `InsufficientHumanData` when there are fewer than
`min_human_examples` (default 300) guesses with text. Align raises
`ProxyNotTrained` unless its base checkpoint came out of a proxy run. The
device defaults to cuda, then mps, then cpu.

## Sizes and measured speed

Default `ModelConfig` (measured with `ModelBundle.param_counts()`):

| network | params |
| --- | ---: |
| Speaker (d=384, 2 enc + 3 dec layers, 6 strokes) | 10,738,616 |
| Listener (conv 48-96-192-320, hidden 512) | 5,118,186 |
| HumanProxy (same encoder → 384-d) | 5,203,344 |
| **total** | **21,060,146** |

Per-step times from `estimate_runtime` (3 probe steps, batch 64) on the
owner's Apple-silicon MacBook. These are single measurements, not
benchmarks:

| stage | default steps | cpu | mps |
| --- | ---: | --- | --- |
| bootstrap | 30,000 | 0.88 s/step, ~7h20m | 0.26 s/step, ~2h10m |
| proxy | 3,000 | 0.60 s/step, ~30m | 0.10 s/step, ~5m |
| align | 5,000 | 1.18 s/step, ~1h40m | 0.25 s/step, ~21m |

## What is not done yet

- **No real training run has happened.** The pipeline has only been run at
  tiny scale in tests, plus one small sanity run: a 4-stroke, roughly
  100k-parameter model, 600 bootstrap steps on CPU, where candidate accuracy
  rose from chance (0.125) to 0.39. Whether the default config reaches high
  accuracy and a meaningful topsim within 30k steps is **unverified**. Expect
  to tune `lr`, `n_distractors`, `ink_weight` and `listener_reset_every` on
  the first real run.
- The served model is the untrained `init.pt` until someone runs bootstrap and
  promotes the result.
- The scoring FLOOR/CEIL constants and the proxy and align defaults
  (`min_human_examples=300`, step counts, `align_game_weight`) are reasoned
  guesses. They need revisiting once real guess data exists.
- No automatic promotion or scheduling exists on purpose. Promotion should be
  a human decision made after reading the metrics.
