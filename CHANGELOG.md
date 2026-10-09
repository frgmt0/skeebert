# Changelog

## 0.1.0 (2026-10-08)

First release of the application layer, on top of the ML core.

- Discord bot (`skeebert bot`): talk by @mention, DM or `/skeebert say`; replies
  are a glyph image with a persistent **Guess** button (survives restarts).
  Guess via the button (private result: tier, score bar, reveal, guess count)
  or by replying to a glyph (tier emoji reaction only). One guess per person
  per glyph; the glyph message shows "decoded by N".
- `/skeebert dictionary` (contact sheet of up to 9 cracked intents on the
  served model, ranked by mean score), `/skeebert help`, `/skeebert privacy`
  (stop/start keeping, delete everything with confirmation).
- One-line first-time notice linking the privacy page; two one-off tips on a
  person's 1st and 3rd guesses.
- Brain: Claude Haiku (`claude-haiku-5-5`, configurable, unverified live) with
  structured output restricted to the atom vocabulary and a cached system
  prompt; persistent mood; up to six messages of in-memory context.
- Falls asleep (local embedding brain + 💤) when the daily budget ($6.00/UTC
  day, from real usage, persisted) is spent, on API errors or bad output, or
  past 30 brain calls per person per hour.
- SQLite store with salted-hash identities, numbered migrations, cascade
  delete, training-example export with the exact shown glyph.
- CLI: `play`, `render`, `summary`, `dictionary`, `spend`, `checkpoints`,
  `promote`, `train` (estimate first, refuses without `--yes`), `forget`,
  `keeping`.
- README, CLAUDE.md and `.env.example`.
- Deployment config (`deploy.toml`) for `deployer` to host `desktop`, with state kept outside releases and secrets from `.env.prod`.

Fixes from review (before release):

- Stop keeping / Delete everything pressed while the brain is thinking (or a
  guess is being scored) now wins: nothing from that message is stored or
  used as context. The store re-checks keeping inside the insert transaction.
- Other people's mentions, role/channel mentions and custom emoji ids are
  replaced before text is stored or sent to Anthropic.
- A first-ever reply guess gets the one-line privacy notice as its own reply.
- The dictionary only quotes phrases at least two people guessed.
- Deleting no longer resets the hourly rate limit, and resets the mood.
- A missing salt next to existing data is refused instead of regenerated;
  `forget` / `keeping` refuse when the data isn't found.
- A guess on a glyph deleted mid-flight reports "gone" instead of erroring;
  interaction and message failures are reported to the user and logged.
- Spend: no hidden SDK retries, timeouts charged at worst case, byte-based
  token bound, ledger write retried and never lost silently.
- Counter edits on one glyph message are serialised; reply guesses are
  clipped to 200 characters; the embedder warm-up task is kept and logged.
- Site wording: replying gets a reaction; the dictionary note in the privacy policy.
