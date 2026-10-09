# skeebert site

Static site for skeebert.frgmt.xyz, served by a Cloudflare Worker with static assets (no worker code, no KV, no API).
Pages live in `public/`; clean URLs (`/privacy` -> `privacy.html`) come from the assets `html_handling` setting.

The home page is a frgmt research release page (the boop26 house style, same family as nano.frgmt.xyz): numbered sections, a benchmark section, a technical report, safety, pricing and a model card.

- `index.html`, `privacy.html`, `terms.html`, `aup.html`: the pages. The policy pages are binding; change their wording only on purpose.
- `style.css`: shared styles. The palette comes from the glyph renderer (`skeebert/render.py`). Light and dark follow the system setting; the toggle in the top bar overrides it and remembers the choice in `localStorage` (no cookies).
- `site.js`: vanilla JS, no dependencies. It re-implements the renderer's soft distance field to draw glyphs from their real stroke data as ordered-dithered ASCII (one character per pixel of the model's 64 × 64 training render), runs the decoder in Fig. 1, draws the ASCII fields at the page edges from real glyph coverage, and renders every benchmark table and chart from `bench.json` and `trainlog.json`. Under `prefers-reduced-motion` everything shows its final frame; the edge fields redraw at ~10 fps and pause when the tab is hidden.
- `fonts/`: Bricolage Grotesque and JetBrains Mono (both SIL Open Font License, see `fonts/OFL.txt` and `fonts/OFL-JetBrainsMono.txt`), self-hosted so the site makes no third-party requests.
- `glyphs/`: real glyphs plus `glyphs.json`, all generated. Don't edit by hand.
- `bench.json`, `trainlog.json`: generated benchmark results and training-log summary. Don't edit by hand.

## Refreshing the glyphs

Every glyph on the site is real model output. After promoting a new checkpoint, re-render them from the repo root:

    uv run python scripts/render_site_glyphs.py --checkpoint-dir checkpoints --out site/public/glyphs --label "model {version}"

The script only reads the checkpoint directory (it refuses to run on one with neither `current` nor `init.pt`). It writes a 512px PNG (the exact image Discord shows), a 768px and a 160px WebP per curated glyph, a 384px WebP for each of the persona's worked examples (`PERSONA_EXAMPLES` in `skeebert/brain/haiku.py`, shown on the page as a constructed exchange), and `glyphs.json` with the model version, the `--label` text, the checkpoint's real parameter counts, and every glyph's intent, gloss and stroke parameters. The curated intents are listed in `CURATED` at the top of the script; the first one is the hero glyph.

## Refreshing the benchmarks

    uv run python scripts/bench.py

Loads `checkpoints/init.pt` and `checkpoints/ckpt-6d99ccb9bda1.pt` read-only and runs inference only, on CPU with fixed seeds (about a minute and a half on an M-series MacBook). It reuses `skeebert.train`'s own eval helpers, so its re-run of the end-of-run eval matches the logged numbers. It writes `public/bench.json` (every number, with seeds, n, device, date, checkpoint versions and the git commit) and `public/trainlog.json` (a summary of the bootstrap run's log in `checkpoints/logs/`). Facts that live outside this machine (the live spend ledger, the human-guess count and mean score, the persona-tuning spend) are kept in `REPORTED` at the top of the script with their source; update them there by hand. When a new checkpoint is promoted, change `TRAINED_FILE`.

The roadmap's "Now" entry and the honest-expectations paragraph are prose in `index.html`. Update them by hand when they stop being true.

## Local dev

    cd site
    npx wrangler dev --local

Then open http://localhost:8787. New files under `public/` need a restart of `wrangler dev` to be served.

## Deploy

    cd site && npx wrangler deploy

This publishes the worker `skeebert-site` and attaches the custom domain `skeebert.frgmt.xyz` (see `wrangler.jsonc`).
