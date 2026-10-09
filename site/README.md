# skeebert site

Static site for skeebert.frgmt.xyz, served by a Cloudflare Worker with static assets (no worker code, no KV, no API).
Pages live in `public/`; clean URLs (`/privacy` -> `privacy.html`) come from the assets `html_handling` setting.

- `index.html`, `privacy.html`, `terms.html`, `aup.html`: the pages. The policy pages are binding; change their wording only on purpose.
- `style.css`: shared styles. The palette comes from the glyph renderer (`skeebert/render.py`). Light and dark follow the system setting.
- `site.js`: vanilla JS, no dependencies. Draws the hero glyph from its real stroke data, runs the decode game, fills the spec numbers from the manifest, and handles scroll reveals (all off under `prefers-reduced-motion`).
- `fonts/`: Bricolage Grotesque (SIL Open Font License, see `fonts/OFL.txt`), self-hosted so the site makes no third-party requests.
- `glyphs/`: real glyphs plus `glyphs.json`, all generated. Don't edit by hand.

## Refreshing the glyphs

Every glyph on the site is real model output. After promoting a new checkpoint, re-render them from the repo root:

    uv run python scripts/render_site_glyphs.py --checkpoint-dir checkpoints --out site/public/glyphs --label "model {version}"

The script only reads the checkpoint directory (it refuses to run on one with neither `current` nor `init.pt`). It writes a 512px PNG (the exact image Discord shows), a 768px and a 160px WebP per glyph, and `glyphs.json` with the model version, the `--label` text (`{version}` is filled in), whether the served file is the untrained `init.pt`, the checkpoint's real parameter counts, and each glyph's intent, gloss and stroke parameters. The page reads everything from the manifest, and its copy changes when `untrained` is false. The curated intents are listed in `CURATED` at the top of the script; the first one is the hero glyph.

The roadmap's "Now" entry and the fine-print paragraph say training hasn't started. Update that copy in `index.html` by hand when it stops being true.

## Local dev

    cd site
    npx wrangler dev --local

Then open http://localhost:8787.

## Deploy

    cd site && npx wrangler deploy

This publishes the worker `skeebert-site` and attaches the custom domain `skeebert.frgmt.xyz` (see `wrangler.jsonc`).
