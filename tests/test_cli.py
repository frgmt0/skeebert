import pytest

from skeebert import cli
from skeebert.identity import Pseudonymiser
from skeebert.store import Store
from skeebert.types import Intent


def run(settings, *argv):
    return cli.main(list(argv), settings=settings)


def test_parser_knows_every_command():
    p = cli.build_parser()
    for argv in (["bot"], ["play"], ["render", "food,question"], ["summary"], ["dictionary"], ["spend"],
                 ["checkpoints"], ["promote", "abc"], ["train", "--stage", "bootstrap"],
                 ["forget", "--discord-user-id", "1"], ["keeping", "--discord-user-id", "1", "off"]):
        assert p.parse_args(argv).command == argv[0]
    with pytest.raises(SystemExit):
        p.parse_args(["train", "--stage", "nope"])
    assert set(cli.COMMANDS) == {"bot", "play", "render", "summary", "dictionary", "spend", "checkpoints",
                                 "promote", "train", "forget", "keeping"}


def test_render_png_and_jpeg(settings, engine, tmp_path, capsys):
    out = tmp_path / "x.png"
    assert run(settings, "render", "question,food", "-o", str(out), "--size", "64") == 0
    assert out.read_bytes().startswith(b"\x89PNG")
    assert "food+question" in capsys.readouterr().out
    jpg = tmp_path / "x.jpg"
    assert run(settings, "render", "food", "-o", str(jpg), "--jpeg", "--size", "64") == 0
    assert jpg.read_bytes()[:2] == b"\xff\xd8"
    assert run(settings, "render", "blorp") == 2


def test_train_refuses_without_yes(settings, engine, monkeypatch, capsys):
    import skeebert.train as train

    def boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("train_from_cli called without --yes")

    monkeypatch.setattr(train, "train_from_cli", boom)
    assert run(settings, "train", "--stage", "bootstrap", "--device", "cpu") == 2
    captured = capsys.readouterr()
    assert "estimate: bootstrap on cpu" in captured.out and "--yes" in captured.err
    assert run(settings, "train", "--stage", "bootstrap", "--device", "cpu", "--estimate") == 0
    assert sorted(p.name for p in settings.checkpoint_dir.glob("*.pt")) == ["init.pt"]  # nothing written


def test_train_proxy_refused_for_lack_of_data(settings, engine, capsys):
    assert run(settings, "train", "--stage", "proxy", "--device", "cpu", "--yes") == 2
    assert "needs at least 300" in capsys.readouterr().err


def test_train_with_yes_passes_stored_examples(settings, engine, monkeypatch):
    import skeebert.train as train

    store = Store(settings.db_path)
    intent = Intent.of("food")
    store.add_exchange(exchange_id="aaaaaaaaaaaa", user_hash="u_a", source="cli", message_text="m", intent=intent,
                       glyph=engine.speak(intent), brain_source="local", asleep=True, mood="tired", notice_shown=False)
    store.add_guess(exchange_id="aaaaaaaaaaaa", user_hash="u_b", guess_text="food", score=0.9, via="cli")
    seen = {}

    def record(examples, stage, **kw):
        seen.update(examples=examples, stage=stage, **kw)
        raise train.TrainingDiverged("stop here")

    monkeypatch.setattr(train, "train_from_cli", record)
    with pytest.raises(train.TrainingDiverged):
        run(settings, "train", "--stage", "bootstrap", "--device", "cpu", "--yes")
    assert seen["confirm"] is True and seen["stage"] == "bootstrap"
    assert [e.guess_text for e in seen["examples"]] == ["food"]
    assert seen["checkpoint_dir"] == settings.checkpoint_dir


def test_summary_spend_checkpoints_dictionary(settings, engine, capsys):
    for cmd in (["summary"], ["spend"], ["checkpoints"], ["dictionary"]):
        assert run(settings, *cmd) == 0
    out = capsys.readouterr().out
    assert engine.model_version in out
    assert "nothing has been cracked yet" in out


def test_promote_unknown_version(settings, engine, capsys):
    assert run(settings, "promote", "doesnotexist") == 2
    assert run(settings, "promote", engine.model_version) == 0


def test_forget_and_keeping_by_discord_id(settings, engine, capsys):
    uh = Pseudonymiser.load(settings.salt_path).user("1234")
    store = Store(settings.db_path)
    intent = Intent.of("food")
    store.add_exchange(exchange_id="aaaaaaaaaaaa", user_hash=uh, source="mention", message_text="m", intent=intent,
                       glyph=engine.speak(intent), brain_source="local", asleep=True, mood="tired", notice_shown=True)
    assert run(settings, "keeping", "--discord-user-id", "1234", "off") == 0
    assert not store.is_keeping(uh)
    assert run(settings, "forget", "--discord-user-id", "1234") == 0
    assert store.user_counts(uh) == (0, 0) and not store.is_keeping(uh)
    assert "deleted 1 exchanges" in capsys.readouterr().out


def test_play_loop(settings, engine, monkeypatch, embedder, capsys):
    import skeebert.service as service

    real = service.build_service
    monkeypatch.setattr(service, "build_service", lambda s: real(s, embedder=embedder, engine=engine))
    inputs = iter(["are you hungry?", "food", "hello"])

    def fake_input(prompt):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    opened = []
    args = cli.build_parser().parse_args(["play"])
    assert cli.cmd_play(settings, args, input_fn=fake_input, opener=opened.append) == 0
    assert len(opened) == 2 and opened[0].exists() and opened[0].parent == settings.renders_dir
    out = capsys.readouterr().out
    assert "skeebert meant:" in out
    store = Store(settings.db_path)
    s = store.summary()
    assert s["exchanges"] == 2 and s["guesses"] == 1


def test_bot_requires_token(settings, capsys):
    assert run(settings, "bot") == 2
    assert "SKEEBERT_DISCORD_TOKEN" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["forget", "--discord-user-id", "1"], ["keeping", "--discord-user-id", "1", "off"]])
def test_operator_commands_refuse_on_missing_data(settings, capsys, argv):
    assert run(settings, *argv) == 2
    err = capsys.readouterr().err
    assert str(settings.data_dir.resolve()) in err
    assert not settings.db_path.exists() and not settings.salt_path.exists()  # nothing was created


def test_forget_refuses_if_salt_lost(settings, engine):
    Store(settings.db_path).set_keeping("u_x", False)
    assert run(settings, "forget", "--discord-user-id", "1") == 2
    assert not settings.salt_path.exists()
