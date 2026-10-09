# skeebert site

Static site for skeebert.frgmt.xyz, served by a Cloudflare Worker with static assets (no worker code, no KV, no API).
Pages live in `public/`; clean URLs (`/privacy` -> `privacy.html`) come from the assets `html_handling` setting.

## Local dev

    cd site
    npx wrangler dev --local

Then open http://localhost:8787.

## Deploy

    cd site && npx wrangler deploy

This publishes the worker `skeebert-site` and attaches the custom domain `skeebert.frgmt.xyz` (see `wrangler.jsonc`).
