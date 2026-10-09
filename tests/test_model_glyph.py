import json

import pytest
import torch

from skeebert import concepts
from skeebert.glyph import (
    CURRENT_NAME,
    INIT_NAME,
    CheckpointError,
    GlyphEngine,
    list_checkpoints,
    promote,
    save_new_checkpoint,
    serving_checkpoint_path,
)
from skeebert.model import HumanProxy, Listener, ModelBundle, ModelConfig, Speaker, pack_intents
from skeebert.types import Glyph, Intent


def test_default_config_is_about_twenty_million_params():
    counts = ModelBundle.create(ModelConfig.default(), 0).param_counts()
    assert 15_000_000 <= counts["total"] <= 25_000_000


def test_speaker_is_order_invariant_and_shapes(tiny_config):
    torch.manual_seed(0)
    sp = Speaker(tiny_config).eval()
    a = pack_intents([Intent.of("food", "question")])
    ids, mask = a
    flipped = (ids[:, [1, 0, 2]], mask[:, [1, 0, 2]])
    out = sp(*a)
    assert out.shape == (1, tiny_config.n_strokes, 8)
    assert torch.allclose(out, sp(*flipped), atol=1e-5)
    lis, prox = Listener(tiny_config), HumanProxy(tiny_config)
    img = torch.rand(2, tiny_config.image_size, tiny_config.image_size)
    assert lis(img).shape == (2, tiny_config.n_concepts)
    p = prox(img)
    assert p.shape == (2, tiny_config.embed_dim)
    assert torch.allclose(p.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_bundle_round_trip_and_version(tmp_path, tiny_config):
    b = ModelBundle.create(tiny_config, 3)
    v1 = b.version()
    assert ModelBundle.create(tiny_config, 3).version() == v1
    assert ModelBundle.create(tiny_config, 4).version() != v1
    version, path = save_new_checkpoint(b, tmp_path)
    assert version == v1 and path.name == f"ckpt-{v1}.pt"
    assert ModelBundle.from_file(path).version() == v1
    with pytest.raises(CheckpointError):
        save_new_checkpoint(b, tmp_path)


def test_grow_concepts_keeps_old_rows(tiny_config):
    small = ModelConfig.from_dict({**tiny_config.to_dict(), "n_concepts": 10})
    b = ModelBundle.create(small, 0)
    old = b.speaker.atom_emb.weight[:10].clone()
    assert b.grow_concepts(12)
    assert b.speaker.atom_emb.weight.shape[0] == 12 and b.listener.out.out_features == 12
    assert torch.equal(b.speaker.atom_emb.weight[:10], old)
    assert not b.grow_concepts(12)


def test_engine_creates_and_persists_init(tmp_path, tiny_config):
    e1 = GlyphEngine.load(tmp_path, config=tiny_config)
    assert (tmp_path / INIT_NAME).exists() and not (tmp_path / CURRENT_NAME).exists()
    intent = Intent.of("excited", "food")
    g1 = e1.speak(intent)
    e2 = GlyphEngine.load(tmp_path)  # no config: must reuse the persisted init.pt
    assert e2.model_version == e1.model_version
    g2 = e2.speak(intent)
    assert g1 == g2 and g1.png_bytes(64) == g2.png_bytes(64)
    assert g1.model_version == e1.model_version
    assert Glyph.from_json(g1.to_json()) == g1


def test_engine_variation_is_seeded(tmp_path, tiny_config):
    e = GlyphEngine.load(tmp_path, config=tiny_config)
    intent = Intent.of("greeting")
    base = e.speak(intent)
    v7 = e.speak(intent, variation=7)
    assert v7 == e.speak(intent, variation=7)
    assert v7 != base and v7.variation == 7 and base.variation is None
    assert e.speak(Intent.of("no")) != base


def test_engine_refuses_atoms_newer_than_checkpoint(tmp_path, tiny_config):
    small = ModelConfig.from_dict({**tiny_config.to_dict(), "n_concepts": 5})
    e = GlyphEngine.load(tmp_path, config=small)
    e.speak(Intent.from_ids([0, 4]))
    with pytest.raises(CheckpointError):
        e.speak(Intent.of(concepts.CONCEPTS[10].name))


def test_promote_switches_serving_and_validates(tmp_path, tiny_config):
    e = GlyphEngine.load(tmp_path, config=tiny_config)
    other = ModelBundle.create(tiny_config, 99)
    v, path = save_new_checkpoint(other, tmp_path)
    assert serving_checkpoint_path(tmp_path).name == INIT_NAME
    promote(tmp_path, v)
    assert (tmp_path / CURRENT_NAME).read_text().strip() == path.name
    assert GlyphEngine.load(tmp_path).model_version == v
    listed = {c["file"]: c for c in list_checkpoints(tmp_path)}
    assert listed[path.name]["serving"] and not listed[INIT_NAME]["serving"]
    with pytest.raises(CheckpointError):
        promote(tmp_path, "doesnotexist")
    promote(tmp_path, e.model_version)  # rolling back to init by its version works
    assert serving_checkpoint_path(tmp_path).name == INIT_NAME
