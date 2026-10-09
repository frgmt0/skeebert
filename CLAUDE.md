# CLAUDE.md: working on skeebert

Skeebert is a Discord bot that answers in glyphs drawn by a small model trained
from scratch, with Claude Haiku as its brain. `README.md` explains how it runs
and is operated; `docs/ML.md` explains the model and training. This file is
the operational map: what must never be broken.

## Where things live

| path | holds | notes |
| --- | --- | --- |
| `data/skeebert.db` | real people's messages, guesses, spend, mood | SQLite, WAL. Never commit. |
| `data/.salt` | the hash salt | never change, never lose, never commit |
| `checkpoints/` | `init.pt`, `ckpt-<version>.pt`, `current`, `logs/` | never commit |
| `renders/` | images from `skeebert render` / `play` | scratch |
| `.env` | Discord token, Anthropic key | never commit, never paste values anywhere |
| `site/` | the public site (skeebert.frgmt.xyz) | its promises are binding |

## Deployment

`deploy.toml` ships to host `desktop` with `deployer` (`deployer`, `deployer
status`, `deployer logs`, `deployer restart`). Secrets come from a local,
gitignored `.env.prod` (made from `.env.example`). Production state lives
outside the release dirs in `/home/jason/.local/share/skeebert/{data,checkpoints,renders}`
(set via `[env]`); `data/` holds the live database and salt, never delete or
recreate it. The active release is `~/.local/share/deployer/skeebert/current`.
Checkpoints are rsynced there by hand, then promoted on desktop with
`skeebert promote VERSION` run from `current` with the three `SKEEBERT_*_DIR`
vars set, then `deployer restart` (see README "Deploying"). Don't deploy,
promote or restart unprompted.

## Invariants

- **The site's promises are binding.** `site/public/privacy.html`, `terms.html`
  and `aup.html` are live. Any change to what is kept, sent to Anthropic,
  shown, or deletable must update the site and the code together, in the same
  change, and anything that widens collection must be announced before it
  applies. Read `privacy.html` before touching `service.py`, `store.py`,
  `identity.py` or `bot.py`.
- **Never change or lose the salt** (`data/.salt`). `forget` and the Delete
  button find people's rows by re-deriving their hash; a new salt orphans
  every row it needs to delete. Back it up with the database.
- **No raw Discord ids in the database or sent to Anthropic.** Ids go through
  `identity.Pseudonymiser` (users `u_…`, glyph messages `m_…`), and all
  message/guess text passes `bot.clean_text` (other mentions, roles,
  channels, custom emoji) before it reaches the service.
- **A salt is only created next to an empty database.** If `data/.salt` is
  missing while data exists, everything refuses to start (`SaltMissing`).
- **Only addressed messages.** The bot processes a message only if it is a DM,
  mentions Skeebert, or replies to a Skeebert glyph message, and never from a
  bot. Don't add listeners that read anything else.
- **Stop keeping means nothing new is stored.** For a person with
  `keeping = 0`, nothing derived from their messages or guesses may reach
  SQLite or the context buffer (the service keeps those exchanges in an
  in-memory LRU only). Stop/delete pressed during a brain call wins: the
  service compares a per-user epoch before/after the call, and the store
  re-checks keeping inside the insert transaction. Keep both checks. The
  keeping flag survives "Delete everything"; the rate limit is not reset by it.
- **`concepts.py` is append-only.** An atom's id is its position. Never
  reorder, rename or delete; only append. A test pins the existing names.
- **Render constants are frozen once glyphs are public.** `render.py`'s
  `COORD_LIMIT`, half-width limits and colours aren't part of the model
  version; changing them silently changes every stored glyph's picture.
- **Checkpoints are never overwritten**, and training never touches
  `current`. **Promotion is explicit**: only `skeebert promote VERSION`
  changes what is served, after a human has read the metrics. Never automate it.
- **`skeebert train` refuses without `--yes`.** Real training runs take hours;
  don't start one unprompted.
- **Never run fakes, tests or experiments against the live data dir.** Tests
  use `tmp_path`; for manual runs set `SKEEBERT_DATA_DIR` and
  `SKEEBERT_CHECKPOINT_DIR` to a scratch directory. `skeebert play` stores
  real rows (under an operator identity) wherever `SKEEBERT_DATA_DIR` points.
- **The budget must stay honest.** Spend is computed from each response's
  real `usage`. If Anthropic's prices change, update `config.Prices` (and its
  source comment) rather than fudging the cap.
- **Migrations are append-only.** Add a new entry to `store.MIGRATIONS`;
  never edit a released one.

## Things Claude should not do unprompted

- Start the real bot, promote or roll back checkpoints, or run real training.
- Call the real Anthropic API from tests (use `tests/conftest.py`'s
  `FakeAnthropic`).
- Loosen anything in the privacy, hashing or "only addressed messages" paths.
- Commit `data/`, `checkpoints/`, `renders/` or any `.env`.

## Working

```bash
uv sync
uv run pytest -q          # no network, no token, ~10s
uv run skeebert summary   # with SKEEBERT_DATA_DIR pointing where you mean it to
```

All Discord-free behaviour is in `service.py` and is tested there; `bot.py`
stays a thin adapter. Blocking work (brain call, embedding, rendering, SQLite)
runs in `asyncio.to_thread` from the bot.
