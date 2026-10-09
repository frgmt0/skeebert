import json

import pytest
import torch

from skeebert.glyph import CURRENT_NAME, INIT_NAME, GlyphEngine, promote, read_meta
from skeebert.model import multi_hot
from skeebert.scoring import score_guess
from skeebert.train import (
    InsufficientHumanData,
    ProxyNotTrained,
    TrainConfig,
    TrainingNotConfirmed,
    estimate_runtime,
    game_loss,
    make_distractors,
    run_training,
    sample_intents,
    select_device,
    topographic_similarity,
    train_from_cli,
)
from skeebert.types import Intent, TrainingExample

GUESSES = ["food", "are you hungry", "hello there", "a question", "big fire", "no idea", "tired", "happy food"]


def _examples(engine, embedder, n=12):
    names = ["food", "question", "greeting", "fire", "big", "no", "tired", "happy", "water", "sad"]
    out = []
    for i in range(n):
        intent = Intent.of(names[i % len(names)], names[(i * 3 + 1) % len(names)]) if i % 2 else Intent.of(names[i % len(names)])
        guess = GUESSES[i % len(GUESSES)]
        out.append(TrainingExample(intent, engine.speak(intent), guess, score_guess(intent, guess, embedder)))
    return out


@pytest.fixture
def ckdir(tmp_path, tiny_config):
    GlyphEngine.load(tmp_path, config=tiny_config)
    return tmp_path


def _files(d):
    return sorted(p.name for p in d.glob("*.pt"))


def _check_run(report, d, before, stage):
    assert torch.isfinite(torch.tensor(report.final_loss))
    assert report.stage == stage and report.version != report.parent_version
    new = set(_files(d)) - set(before)
    assert new == {f"ckpt-{report.version}.pt"}
    assert not (d / CURRENT_NAME).exists()  # training never promotes
    lines = [json.loads(l) for l in open(report.log_path)]
    assert lines[0]["event"] == "start" and lines[-1]["event"] == "done"
    assert any(l["event"] == "eta" for l in lines)
    steps = [l for l in lines if l["event"] == "step"]
    assert steps and all(torch.isfinite(torch.tensor(l["loss"])) for l in steps)
    meta = read_meta(report.checkpoint_path)
    assert meta["stage"] == stage and meta["parent"] == report.parent_version and meta["version"] == report.version


def test_all_three_stages_chain(ckdir, embedder):
    init_bytes = (ckdir / INIT_NAME).read_bytes()
    before = _files(ckdir)
    boot = run_training([], "bootstrap", TrainConfig.tiny(steps=4), ckdir, "cpu")
    _check_run(boot, ckdir, before, "bootstrap")
    assert "eval_topsim" in boot.metrics and boot.metrics["listener_resets"] == 1

    engine = GlyphEngine.load(ckdir)
    examples = _examples(engine, embedder)
    before = _files(ckdir)
    prox = run_training(examples, "proxy", TrainConfig.tiny(), ckdir, "cpu", base=boot.version, embedder=embedder)
    _check_run(prox, ckdir, before, "proxy")
    assert prox.parent_version == boot.version
    assert read_meta(prox.checkpoint_path)["proxy_trained"] is True
    assert prox.metrics["n_val"] == 1 and "val_baseline_cos" in prox.metrics

    before = _files(ckdir)
    al = run_training(examples, "align", TrainConfig.tiny(), ckdir, "cpu", base=prox.version, embedder=embedder)
    _check_run(al, ckdir, before, "align")
    assert "eval_align_cos" in al.metrics
    assert read_meta(al.checkpoint_path)["proxy_trained"] is True

    # init.pt untouched, and nothing served until an explicit promote
    assert (ckdir / INIT_NAME).read_bytes() == init_bytes
    assert GlyphEngine.load(ckdir).model_version == engine.model_version
    promote(ckdir, al.version)
    assert GlyphEngine.load(ckdir).model_version == al.version


def test_proxy_and_align_refuse_without_enough_human_data(ckdir, embedder):
    engine = GlyphEngine.load(ckdir)
    few = _examples(engine, embedder, n=4)
    for stage in ("proxy", "align"):
        with pytest.raises(InsufficientHumanData):
            run_training(few, stage, TrainConfig.tiny(min_human_examples=5), ckdir, "cpu", embedder=embedder)
    blank = [TrainingExample(e.intent, e.glyph, "  ", 0.0) for e in _examples(engine, embedder, n=10)]
    with pytest.raises(InsufficientHumanData):
        run_training(blank, "proxy", TrainConfig.tiny(min_human_examples=5), ckdir, "cpu", embedder=embedder)
    assert _files(ckdir) == [INIT_NAME]


def test_align_refuses_untrained_proxy(ckdir, embedder):
    examples = _examples(GlyphEngine.load(ckdir), embedder)
    with pytest.raises(ProxyNotTrained):
        run_training(examples, "align", TrainConfig.tiny(), ckdir, "cpu", embedder=embedder)


def test_cli_entry_requires_confirmation(ckdir):
    with pytest.raises(TrainingNotConfirmed):
        train_from_cli([], "bootstrap", confirm=False, config=TrainConfig.tiny(), checkpoint_dir=ckdir, device="cpu")
    assert _files(ckdir) == [INIT_NAME]
    report = train_from_cli([], "bootstrap", confirm=True, config=TrainConfig.tiny(steps=1), checkpoint_dir=ckdir, device="cpu")
    assert report.checkpoint_path.endswith(f"ckpt-{report.version}.pt")


def test_unknown_stage_and_estimate(ckdir, tmp_path):
    with pytest.raises(ValueError):
        run_training([], "dance", TrainConfig.tiny(), ckdir, "cpu")
    est = estimate_runtime([], "bootstrap", TrainConfig.tiny(steps=100), ckdir, "cpu", probe_steps=1)
    assert est.seconds_per_step > 0 and est.total_seconds > est.seconds_per_step * 99
    assert "bootstrap" in est.describe()
    assert _files(ckdir) == [INIT_NAME]  # estimating writes nothing
    assert not (ckdir / "logs").exists()


def test_select_device():
    assert select_device("cpu").type == "cpu"
    assert select_device("auto").type in {"cpu", "cuda", "mps"}


def test_distractors_and_game_loss():
    gen = torch.Generator().manual_seed(0)
    ids, mask = sample_intents(32, 20, (0.4, 0.35, 0.25), gen)
    target = multi_hot(ids, mask, 20)
    assert ((target.sum(-1) >= 1) & (target.sum(-1) <= 3)).all()
    cands = make_distractors(ids, mask, 5, 20, 1.0, (0.4, 0.35, 0.25), gen)
    assert cands.shape == (32, 5, 20)
    sizes = cands.sum(-1)
    assert ((sizes >= 1) & (sizes <= 3)).all()
    # hard distractors differ from the target by at most one swap
    diff = (cands - target.unsqueeze(1)).abs().sum(-1)
    assert (diff <= 2).all()
    # a listener that outputs the target exactly wins the game
    perfect = (target * 2 - 1) * 20
    ce, bce, acc = game_loss(perfect, target, cands)
    assert acc.item() == 1.0 and ce.item() < 1e-3


def test_topographic_similarity_detects_structure():
    gen = torch.Generator().manual_seed(1)
    ids, mask = sample_intents(40, 8, (0.4, 0.35, 0.25), gen)
    target = multi_hot(ids, mask, 8)
    # compositional "images": each atom lights its own pixel block
    comp = target.repeat_interleave(16, dim=1).view(40, 8, 16)
    random_imgs = torch.rand(40, 8, 16, generator=gen)
    assert topographic_similarity(target, comp) > 0.5
    assert abs(topographic_similarity(target, random_imgs)) < 0.3
