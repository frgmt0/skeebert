# skeebert

A small alien that talks in glyphs. Skeebert is a Discord bot: you say something
to it, it answers with a little stroke picture, and people guess what the
picture means. Nobody knows the language yet, Skeebert included. The guesses
are the training data from which a shared picture language is supposed to
emerge.

Public site, with the binding privacy policy, terms and acceptable use:
<https://skeebert.frgmt.xyz>.

Skeebert has two halves:

- **the brain** (Claude Haiku via Anthropic's API) decides *what* to say, as
  1–3 concepts ("atoms") from a fixed vocabulary of about 200;
- **the mouth** (`GlyphEngine`, ~21M params, trained from scratch here) turns
  those concepts into a glyph. The brain never sees or picks shapes.

How the mouth is trained is in [`docs/ML.md`](docs/ML.md). This README covers
running the bot and operating it.

## Quickstart

```bash
uv sync                                  # installs everything (CPU torch on Linux)
cp .env.example .env                     # then fill in tokens (see below)
uv run skeebert render food,question     # draws a glyph into renders/
uv run skeebert play                     # talk to it in the terminal and guess
uv run skeebert bot                      # run the Discord bot
uv run pytest -q                         # tests: no token, no network
```

The first run creates `checkpoints/init.pt` (the seeded, untrained mouth) and
`data/.salt`. Until someone trains and promotes a checkpoint, the glyphs are
noise; that's expected.

`skeebert play` works without any key: without `ANTHROPIC_API_KEY` it uses the
local (sleeping) brain. Its exchanges are stored like any other under an
operator identity, with source `cli`, so point `SKEEBERT_DATA_DIR` somewhere
else if you're just experimenting.

## Discord setup

1. Create an application at <https://discord.com/developers/applications>.
2. **Bot** tab: reset and copy the token into `.env` as
   `SKEEBERT_DISCORD_TOKEN`. Turn on **Message Content Intent**
   (privileged). Without it Discord still delivers DMs and messages that
   mention Skeebert, but plain replies to a glyph arrive empty and are
   ignored, so reply-guessing won't work.
3. **OAuth2 → URL Generator**: scopes `bot` and `applications.commands`;
   permissions *View Channels*, *Send Messages*, *Send Messages in Threads*,
   *Attach Files*, *Add Reactions*, *Read Message History*. Open the URL to
   invite the bot.
4. `uv run skeebert bot`. Slash commands sync globally on start, which can
   take a while to show up. For development set `SKEEBERT_DEV_GUILD_ID` to
   a test server id and they sync to that server instantly.

Only run one bot process per token.

## How a turn works

1. Someone addresses Skeebert: `@skeebert …` in a server, any DM, or
   `/skeebert say`. Nothing else is read: messages that don't mention it, and
   anything from bots, are ignored before their content is looked at.
2. The brain gets the message, up to six recent Skeebert-addressed messages
   from the same channel/DM (an in-memory ring buffer that is never written
   to disk), and Skeebert's current mood (a feeling atom kept in SQLite).
   It returns an intent (1–3 atoms) and a new mood through structured output
   whose fields are enums of the atom names.
3. The mouth draws the intent. The reply is the glyph PNG with a **Guess**
   button and almost no text: 💤 if the brain was asleep, and, the very first
   time a person talks to Skeebert, one line linking the privacy page.
4. **Guess button** → a one-field modal → a private result: a closeness word
   (🎯 nailed it / 🔥 close / 🤏 warm / 🧊 cold), a score bar, what Skeebert
   actually meant, and how many people have guessed it. A person's 1st and
   3rd guesses carry one short tip each; never again after that.
5. **Replying to a glyph** with a guess also counts. Skeebert only reacts with
   the tier emoji so the meaning isn't spoiled for others; the button shows
   the reveal. If that reply is the person's first interaction ever, Skeebert
   also replies with just the one-line privacy notice (no reveal). Replies
   are clipped to 200 characters, like the modal.
6. One guess per person per glyph counts. Pressing Guess again shows your
   result; later replies are ignored. The glyph message updates to
   "decoded by N".

Scoring (`skeebert.scoring`) compares the guess with the meaning by sentence
embedding (all-MiniLM-L6-v2, loaded on first use) and gives partial credit per
atom.

## Commands

Discord (`/skeebert …`):

| command | does |
| --- | --- |
| `say <message>` | talk to Skeebert |
| `dictionary` | intents on the served model with ≥3 guesses, best mean score first: a numbered contact sheet of up to 9 glyphs plus "1 · food + question — avg 0.71 over 5 guesses — people said: …". Only phrases at least two different people guessed are quoted (normalised, clipped to 40 characters, never attributed) |
| `help` | five lines and a link |
| `privacy` | private: whether your stuff is kept, how many messages/guesses are stored, and buttons **Stop keeping** / **Start keeping again** and **Delete everything** (with a confirm step) |

Terminal (`uv run skeebert …`):

| command | does |
| --- | --- |
| `bot` | run the Discord bot |
| `play [--no-open]` | type a message, see the glyph (saved in `renders/`, opened on macOS), type a guess, get the score and reveal |
| `render food,question [-o path] [--jpeg] [--size N]` | draw an intent with the served model |
| `summary` | stored counts, served model, mood, today's spend |
| `dictionary [-o sheet.png]` | the same dictionary the bot shows |
| `spend [--days N]` | brain spend and tokens per UTC day |
| `checkpoints` | every checkpoint, which one is served |
| `promote VERSION` | serve a checkpoint (restart the bot afterwards) |
| `train --stage {bootstrap,proxy,align} [--base V] [--device D] [--estimate] [--yes]` | see Training below |
| `forget --discord-user-id ID` | delete everything stored for one person |
| `keeping --discord-user-id ID [on\|off]` | show or set whether one person's stuff is kept |

## Data and privacy

The live policy is <https://skeebert.frgmt.xyz/privacy>, and the code must
behave exactly as it says. In short:

- **Kept by default, disclosed up front.** Messages addressed to Skeebert
  (mentions, `/skeebert` commands, DMs), guesses (button or reply), the
  intent, the exact glyph shown and the score. Nothing else: no channel
  history, server/channel names, avatars, member lists or presence.
- **Pseudonymous.** People are stored as a salted SHA-256 of their Discord
  user id; the glyph messages are remembered as salted hashes of their
  message ids. Other people's mentions, roles, channels and custom emoji in a
  message become `@someone` / `@role` / `#channel` / `:name:` before
  anything is stored or sent, so no raw Discord id is written to the
  database or sent to Anthropic. The salt is in `data/.salt` and never
  leaves the machine. A new salt is only ever created next to an empty
  database; if the salt is missing while data exists, the bot and the CLI
  refuse to start rather than orphan every row.
- **Sent to Anthropic.** The message text and the few recent messages used as
  context go to Anthropic's API so the brain can answer. Messages from people
  who stopped keeping are never used as context for others.
- **Not published, never sold.**
- **Self-serve controls.** *Stop keeping*: Skeebert still answers, but nothing
  derived from your messages or guesses is stored. Your exchanges live only in
  an in-memory LRU so other people's guesses can still be scored and
  revealed, and those guesses aren't stored either; your own guesses on other
  people's glyphs aren't stored. The only things kept about you are the
  "not keeping" setting itself and the fact that you've seen the first-time
  notice. *Delete everything*: removes your exchanges (with every guess
  anyone made on them), your guesses, and everything else tied to your hash,
  on disk and in memory, and resets Skeebert's mood to the default. Your
  keeping setting survives a delete; the hourly rate limit does not reset.
  Pressing either button while Skeebert is still thinking about your message
  wins: that message is not stored. Operators can do the same with
  `skeebert forget` / `skeebert keeping`, which refuse to run (and print the
  resolved data path) if the database or salt isn't where they look.
- **The dictionary** shows meanings people cracked and phrases at least two
  people guessed, never who guessed them.
- Deleted data is excluded from all future training runs (training reads only
  what's in the database). Weights that already learned from it can't
  un-learn it.

## Budget and sleep

- The brain costs money per token. Each response's real `usage` (uncached
  input, output, cache reads, cache writes) × the configured prices is added
  to the current UTC day in SQLite, so restarts don't reset it.
- Before each call a reservation for that call's worst case (prompt bounded
  by its UTF-8 byte length, plus full output) is checked against the daily
  cap (default **$6.00/day**) under a lock, so concurrent calls can't
  overshoot it together. The SDK's automatic retries are off, so one
  reservation covers one billed attempt; a timed-out call is charged its
  whole reservation, since it may have been billed. If the ledger write
  fails it is retried, and if it still fails the cost stays counted against
  today's cap in memory and is logged.
- Skeebert **falls asleep** when the cap is hit, when the API errors or
  returns something outside the vocabulary, or when one person exceeds
  **30 brain calls/hour**. Asleep, the local brain answers (nearest atoms by
  embedding, plus `tired`) and the message shows 💤. It wakes up at the next
  UTC day on its own.
- Prices default to Claude Haiku 5.5's launch rates for prompts ≤100K tokens
  (input $0.10, output $0.50, cache read $0.01, cache write $0.125 per
  million tokens). **Verify them on Anthropic's pricing page** and override
  with `SKEEBERT_PRICE_*` if they changed. Requests are kept far under 100K
  tokens (the system prompt is ~9K by the pessimistic byte estimate; history
  and message are trimmed to keep the estimate under 20K).
- The system prompt (persona + atom list) never changes between calls and is
  marked for prompt caching.

### The model id is unverified

The brain defaults to `claude-haiku-5-5` with structured output,
`output_config.effort = "low"` and no `thinking` parameter. **No live call
has been made with this code**: there was no API key on the machine where it
was written. Before relying on it, run `uv run skeebert play` with a key and
check `uv run skeebert spend` shows cache reads after the first call. Change
the model with `SKEEBERT_BRAIN_MODEL`.

## Training and promotion

Training never changes what users see. Only `promote` does.

```bash
uv run skeebert train --stage bootstrap --estimate   # time a few real steps, write nothing
uv run skeebert train --stage bootstrap              # prints the estimate and refuses
uv run skeebert train --stage bootstrap --yes        # runs; writes checkpoints/ckpt-<version>.pt
uv run skeebert checkpoints                          # read the metrics
uv run skeebert promote <version>                    # serve it
# restart the bot
```

`proxy` and `align` train on the stored guesses (`store.training_examples()`,
each with the exact glyph that was shown) and refuse with fewer than 300
guesses with text. `align` needs a proxy-trained base (`--base`). Stage
details, measured speeds and the checkpoint rules are in `docs/ML.md`.

The dictionary only counts glyphs drawn by the currently served model, so it
starts empty again after a promotion.

## Configuration

All settings come from the environment or `.env`; see `.env.example`.
`SKEEBERT_DATA_DIR`, `SKEEBERT_CHECKPOINT_DIR` and `SKEEBERT_RENDERS_DIR` move
the data. Never point experiments at the live `data/`.

## Deploying

Skeebert deploys to the Linux host `desktop` as a systemd user service with
the `deployer` CLI (`deploy.toml`).

```bash
cp .env.example .env.prod   # keep only SKEEBERT_DISCORD_TOKEN, ANTHROPIC_API_KEY and any overrides
deployer                    # ship the working tree (deployer deploy --dry-run to preview)
deployer status
deployer logs
deployer restart
```

`.env.prod` is gitignored and copied over SSH to a mode-600 file on the host,
never into the release. The deployed `[env]` points state outside the
release directories, under `/home/jason/.local/share/skeebert/` on desktop:
`data/` (database and the hash salt: never delete or recreate it),
`checkpoints/` and `renders/`. Code deploys never touch them. The bot only
makes outbound connections, so there is no tunnel.

Checkpoints are trained on the Mac and shipped separately from code:

```bash
rsync -av checkpoints/ desktop:/home/jason/.local/share/skeebert/checkpoints/
ssh desktop 'cd ~/.local/share/deployer/skeebert/current && \
  SKEEBERT_DATA_DIR=/home/jason/.local/share/skeebert/data \
  SKEEBERT_CHECKPOINT_DIR=/home/jason/.local/share/skeebert/checkpoints \
  .venv/bin/skeebert checkpoints'          # read the metrics, then:
ssh desktop 'cd ~/.local/share/deployer/skeebert/current && \
  SKEEBERT_DATA_DIR=/home/jason/.local/share/skeebert/data \
  SKEEBERT_CHECKPOINT_DIR=/home/jason/.local/share/skeebert/checkpoints \
  .venv/bin/skeebert promote VERSION'
deployer restart
```

`current` is deployer's symlink to the active release.

## Layout

| module | is |
| --- | --- |
| `skeebert/config.py` | settings and prices |
| `skeebert/identity.py` | salt and hashing |
| `skeebert/store.py` | SQLite: users, exchanges, guesses, spend, state |
| `skeebert/budget.py` | daily budget and per-user rate limit |
| `skeebert/brain/` | `HaikuBrain` (awake) and `LocalBrain` (asleep) |
| `skeebert/service.py` | everything the bot does, without Discord |
| `skeebert/sheet.py` | dictionary contact sheet |
| `skeebert/bot.py` | the discord.py adapter |
| `skeebert/cli.py` | the `skeebert` command |
| the rest | the ML core, see `docs/ML.md` |
