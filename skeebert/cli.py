"""``skeebert`` command line: run the bot, play locally, and operate data and training.

Every command reads settings from the environment / ``.env`` (see
``config.Settings``). Point ``SKEEBERT_DATA_DIR`` and
``SKEEBERT_CHECKPOINT_DIR`` somewhere else when experimenting: the default
``data/`` is where real people's data lives on the bot's machine.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .config import Settings
from .identity import SaltMissing

OPERATOR_ID = "operator:cli"  # can never collide with a Discord snowflake (those are integers)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="skeebert", description="Skeebert: a small alien that talks in glyphs.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("bot", help="run the Discord bot (needs SKEEBERT_DISCORD_TOKEN)")

    play = sub.add_parser("play", help="talk to skeebert in the terminal and guess its glyphs")
    play.add_argument("--no-open", action="store_true", help="don't open each glyph image automatically")

    render = sub.add_parser("render", help="draw an intent with the served model, e.g. food,question")
    render.add_argument("atoms", help="1-3 comma-separated atom names")
    render.add_argument("-o", "--output", type=Path, help="output file (default renders/<atoms>.png)")
    render.add_argument("--jpeg", action="store_true", help="write JPEG instead of PNG")
    render.add_argument("--size", type=int, default=512)

    sub.add_parser("summary", help="what is stored, what is served, today's spend")

    dic = sub.add_parser("dictionary", help="print the dictionary (intents with >=3 guesses on the served model)")
    dic.add_argument("-o", "--output", type=Path, help="also write the contact sheet PNG here")

    spend = sub.add_parser("spend", help="brain API spend per UTC day")
    spend.add_argument("--days", type=int, default=14)

    sub.add_parser("checkpoints", help="list checkpoints and which one is served")

    promote = sub.add_parser("promote", help="serve a checkpoint version (restart the bot afterwards)")
    promote.add_argument("version")

    train = sub.add_parser("train", help="train a new checkpoint from stored guesses (never promotes)")
    train.add_argument("--stage", required=True, choices=["bootstrap", "proxy", "align"])
    train.add_argument("--base", help="checkpoint version to start from (default: the served one)")
    train.add_argument("--device", help="cuda | mps | cpu (default: best available)")
    train.add_argument("--estimate", action="store_true", help="only time a few steps and print the estimate")
    train.add_argument("--yes", action="store_true", help="actually run training after printing the estimate")

    forget = sub.add_parser("forget", help="delete everything stored for one Discord user")
    forget.add_argument("--discord-user-id", required=True)

    keeping = sub.add_parser("keeping", help="show or set whether a Discord user's stuff is kept")
    keeping.add_argument("--discord-user-id", required=True)
    keeping.add_argument("state", nargs="?", choices=["on", "off"], help="omit to just show the current state")
    return p


# -- helpers -----------------------------------------------------------------

def _store(settings: Settings):
    from .store import Store

    return Store(settings.db_path)


def _served_version(settings: Settings) -> str | None:
    from .glyph import read_meta, serving_checkpoint_path

    path = serving_checkpoint_path(settings.checkpoint_dir)
    if not path.exists():
        return None
    return read_meta(path).get("version")


def _parse_atoms(text: str):
    from .types import Intent

    atoms = [a.strip() for a in text.replace("+", ",").split(",") if a.strip()]
    return Intent(tuple(atoms))


# -- commands ----------------------------------------------------------------

def cmd_bot(settings: Settings, args) -> int:
    if not settings.discord_token:
        print("SKEEBERT_DISCORD_TOKEN is not set (put it in .env). See README: Discord setup.", file=sys.stderr)
        return 2
    from .bot import run
    from .service import build_service

    service = build_service(settings)
    brain = type(service.brain).__name__
    print(f"serving model {service.model_version} with {brain}; data in {settings.data_dir}")
    run(service, settings.discord_token, settings.dev_guild_id)
    return 0


def cmd_play(settings: Settings, args, input_fn=input, opener=None) -> int:
    from .service import ExchangeGone, build_service

    service = build_service(settings)
    brain = "Claude Haiku" if type(service.brain).__name__ == "HaikuBrain" else "the local (sleeping) brain"
    print(f"talking to skeebert ({brain}, model {service.model_version}). ctrl-d to stop.")
    settings.renders_dir.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            text = input_fn("you> ")
        except EOFError:
            print()
            return 0
        reply = service.handle_talk(OPERATOR_ID, text, source="cli", conversation_key="cli")
        path = settings.renders_dir / f"{reply.exchange_id}.png"
        path.write_bytes(reply.png)
        marker = " 💤 (asleep)" if reply.thought.asleep else ""
        print(f"skeebert drew {path}{marker}")
        if not args.no_open:
            (opener or _open)(path)
        try:
            guess = input_fn("your guess> ")
        except EOFError:
            print()
            return 0
        try:
            outcome = service.handle_guess(OPERATOR_ID, reply.exchange_id, guess, via="cli")
        except ExchangeGone:
            print("that exchange is gone")
            continue
        print(outcome.render().replace("**", ""))


def _open(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        print(f"(open {path} to see it)")


def cmd_render(settings: Settings, args) -> int:
    from .glyph import GlyphEngine

    intent = _parse_atoms(args.atoms)
    engine = GlyphEngine.load(settings.checkpoint_dir, device=settings.device)
    glyph = engine.speak(intent)
    ext = "jpg" if args.jpeg else "png"
    out = args.output or settings.renders_dir / f"{'+'.join(intent.concepts)}.{ext}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(glyph.jpeg_bytes(args.size) if args.jpeg else glyph.png_bytes(args.size))
    print(f"{intent} ({intent.gloss()}) -> {out} [model {glyph.model_version}]")
    return 0


def cmd_summary(settings: Settings, args) -> int:
    from .budget import utc_day, utc_now
    from .glyph import list_checkpoints
    from .service import MOOD_KEY

    store = _store(settings)
    s = store.summary()
    served = _served_version(settings)
    day = utc_day(utc_now())
    spent = store.spent_on(day)
    print(f"data dir:        {settings.data_dir}")
    print(f"users:           {s['users']} ({s['users_not_keeping']} not keeping)")
    print(f"exchanges:       {s['exchanges']} ({s['exchanges_asleep']} while asleep), {s['distinct_intents']} distinct intents")
    mean = "-" if s["mean_score"] is None else f"{s['mean_score']:.3f}"
    print(f"guesses:         {s['guesses']} ({s['guesses_with_text']} with text), mean score {mean}")
    for version, n in s["model_versions"]:
        print(f"  drawn by {version}: {n}")
    print(f"mood:            {store.get_state(MOOD_KEY, 'curious')}")
    print(f"served model:    {served or 'none yet (init.pt is created on first use)'}")
    print(f"checkpoints:     {len(list_checkpoints(settings.checkpoint_dir))} in {settings.checkpoint_dir}")
    print(f"brain:           {settings.brain_model if settings.anthropic_api_key else 'local only (no ANTHROPIC_API_KEY)'}")
    state = "asleep" if spent >= settings.daily_budget_usd else "awake"
    print(f"spend today:     ${spent:.4f} of ${settings.daily_budget_usd:.2f} ({day} UTC, {state})")
    return 0


def cmd_dictionary(settings: Settings, args) -> int:
    from .service import build_dictionary

    served = _served_version(settings)
    if served is None:
        print("no checkpoint is served yet, so nothing has been drawn.")
        return 0
    d = build_dictionary(_store(settings), served)
    print(f"model {served}")
    print(d.render())
    if args.output and d.png is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(d.png)
        print(f"contact sheet -> {args.output}")
    return 0


def cmd_spend(settings: Settings, args) -> int:
    days = _store(settings).spend_days(args.days)
    print(f"daily cap ${settings.daily_budget_usd:.2f} (UTC days); prices per MTok: in ${settings.prices.input}, "
          f"out ${settings.prices.output}, cache read ${settings.prices.cache_read}, cache write ${settings.prices.cache_write}")
    if not days:
        print("no brain spend recorded yet.")
        return 0
    print(f"{'day':<12}{'usd':>10}{'calls':>7}{'input':>10}{'output':>9}{'c.read':>10}{'c.write':>9}")
    for d in days:
        print(f"{d.day:<12}{d.usd:>10.4f}{d.calls:>7}{d.input_tokens:>10}{d.output_tokens:>9}"
              f"{d.cache_read_tokens:>10}{d.cache_write_tokens:>9}")
    return 0


def cmd_checkpoints(settings: Settings, args) -> int:
    from .glyph import list_checkpoints

    rows = list_checkpoints(settings.checkpoint_dir)
    if not rows:
        print(f"no checkpoints in {settings.checkpoint_dir} (init.pt is created on first use)")
        return 0
    for r in rows:
        mark = "*" if r["serving"] else " "
        metrics = r.get("metrics") or {}
        brief = ", ".join(f"{k}={v:.3g}" for k, v in list(metrics.items())[:4])
        print(f"{mark} {r['version']}  {r['file']:<24} stage={r.get('stage', 'init')} parent={r.get('parent') or '-'} {brief}")
    print("* = served")
    return 0


def cmd_promote(settings: Settings, args) -> int:
    from .glyph import CheckpointError, promote

    try:
        path = promote(settings.checkpoint_dir, args.version)
    except CheckpointError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"now serving {path.name}. restart the bot to pick it up.")
    return 0


def cmd_train(settings: Settings, args) -> int:
    from .train import TrainingRefused, estimate_runtime, train_from_cli

    examples = _store(settings).training_examples()
    print(f"{len(examples)} stored guesses available as training examples")
    try:
        est = estimate_runtime(examples, args.stage, checkpoint_dir=settings.checkpoint_dir, device=args.device, base=args.base)
    except TrainingRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(f"estimate: {est.describe()}")
    if args.estimate:
        return 0
    if not args.yes:
        print("not starting: re-run with --yes to train (training never changes what is served).", file=sys.stderr)
        return 2

    def progress(rec: dict) -> None:
        if rec.get("event") in ("eta", "step", "done"):
            print({k: v for k, v in rec.items() if k not in ("config", "params")}, flush=True)

    try:
        report = train_from_cli(
            examples, args.stage, confirm=True, checkpoint_dir=settings.checkpoint_dir, device=args.device,
            base=args.base, progress=progress,
        )
    except TrainingRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(f"new checkpoint {report.version} ({report.checkpoint_path}); metrics: {report.metrics}")
    print(f"not promoted. to serve it: skeebert promote {report.version}")
    return 0


def _existing_identity(settings: Settings):
    """(store, pseudonymiser) for operator commands on real people's data, or None after explaining why.

    Refuses to create anything: run from the wrong directory and these
    commands would otherwise mint a fresh salt and database, hash the id with
    the wrong salt, and report success while deleting nothing.
    """
    from .identity import Pseudonymiser

    db, salt = settings.db_path.resolve(), settings.salt_path.resolve()
    missing = [str(p) for p in (db, salt) if not p.exists()]
    if missing:
        print(f"refusing: {', '.join(missing)} not found (data dir resolves to {settings.data_dir.resolve()}). "
              "Run from the bot's directory or set SKEEBERT_DATA_DIR.", file=sys.stderr)
        return None
    return _store(settings), Pseudonymiser.load(salt, create=False)


def cmd_forget(settings: Settings, args) -> int:
    found = _existing_identity(settings)
    if found is None:
        return 2
    store, ident = found
    uh = ident.user(args.discord_user_id)
    r = store.forget(uh)
    print(f"data dir {settings.data_dir.resolve()}")
    print(f"{uh}: deleted {r.exchanges} exchanges (+{r.guesses_on_their_exchanges} guesses on them) and "
          f"{r.their_guesses} of their guesses; mood reset. A running bot also holds in-memory copies of "
          "not-kept exchanges and recent context until it restarts; the person's own Delete button purges "
          "those immediately.")
    return 0


def cmd_keeping(settings: Settings, args) -> int:
    found = _existing_identity(settings)
    if found is None:
        return 2
    store, ident = found
    uh = ident.user(args.discord_user_id)
    if args.state is not None:
        store.set_keeping(uh, args.state == "on")
    ex, gu = store.user_counts(uh)
    print(f"data dir {settings.data_dir.resolve()}")
    print(f"{uh}: keeping={'on' if store.is_keeping(uh) else 'off'}, stored messages={ex}, guesses={gu}")
    return 0


COMMANDS = {
    "bot": cmd_bot, "play": cmd_play, "render": cmd_render, "summary": cmd_summary, "dictionary": cmd_dictionary,
    "spend": cmd_spend, "checkpoints": cmd_checkpoints, "promote": cmd_promote, "train": cmd_train,
    "forget": cmd_forget, "keeping": cmd_keeping,
}


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = settings or Settings.from_env()
    try:
        return COMMANDS[args.command](settings, args)
    except (ValueError, SaltMissing) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
