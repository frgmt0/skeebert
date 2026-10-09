import os
import stat

import pytest

from skeebert.config import Settings
from skeebert.identity import Pseudonymiser


def test_salt_created_once_and_stable(tmp_path):
    path = tmp_path / "data" / ".salt"
    a = Pseudonymiser.load(path)
    salt = path.read_text()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    b = Pseudonymiser.load(path)
    assert path.read_text() == salt  # never rewritten
    assert a.user(1234) == b.user(1234)
    assert a.user(1234).startswith("u_") and len(a.user(1234)) == 18
    assert a.message(99).startswith("m_")


def test_hash_depends_on_salt_and_id(tmp_path):
    a = Pseudonymiser.load(tmp_path / "a")
    b = Pseudonymiser.load(tmp_path / "b")
    assert a.user(1) != b.user(1)
    assert a.user(1) != a.user(2)
    assert "1234" not in a.user(1234)
    assert a.user(5) != a.message(5)  # namespaces differ


def test_empty_salt_file_is_an_error(tmp_path):
    path = tmp_path / ".salt"
    path.write_text("")
    with pytest.raises(ValueError):
        Pseudonymiser.load(path)
    with pytest.raises(ValueError):
        Pseudonymiser("")


def test_settings_from_env(tmp_path):
    env = {
        "SKEEBERT_DATA_DIR": str(tmp_path / "d"),
        "SKEEBERT_DAILY_BUDGET_USD": "2.5",
        "SKEEBERT_BRAIN_MODEL": "claude-test",
        "SKEEBERT_RATE_LIMIT_PER_HOUR": "7",
        "SKEEBERT_PRICE_OUTPUT_PER_MTOK": "0.6",
        "ANTHROPIC_API_KEY": "sk-test",
    }
    s = Settings.from_env(env, env_file=None)
    assert s.data_dir == tmp_path / "d"
    assert s.db_path == tmp_path / "d" / "skeebert.db"
    assert s.salt_path == tmp_path / "d" / ".salt"
    assert s.daily_budget_usd == 2.5
    assert s.brain_model == "claude-test"
    assert s.rate_limit_per_hour == 7
    assert s.prices.output == 0.6 and s.prices.input == 0.10
    assert s.anthropic_api_key == "sk-test"
    assert s.privacy_url == "https://skeebert.frgmt.xyz/privacy"


def test_settings_defaults_and_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("SKEEBERT_DAILY_BUDGET_USD=1.25\nSKEEBERT_BRAIN_MODEL=from-file\n")
    s = Settings.from_env({"SKEEBERT_BRAIN_MODEL": "from-env"}, env_file=env_file)
    assert s.daily_budget_usd == 1.25
    assert s.brain_model == "from-env"  # real env wins over .env
    d = Settings.from_env({}, env_file=None)
    assert d.brain_model == "claude-haiku-5-5"
    assert d.daily_budget_usd == 6.0
    assert d.rate_limit_per_hour == 30
    assert d.brain_max_input_tokens < 100_000


def test_settings_rejects_bad_numbers():
    with pytest.raises(ValueError):
        Settings.from_env({"SKEEBERT_DAILY_BUDGET_USD": "lots"}, env_file=None)
    with pytest.raises(ValueError):
        Settings.from_env({"SKEEBERT_BRAIN_EFFORT": "extreme"}, env_file=None)


def test_refuses_to_create_salt_when_told_not_to(tmp_path):
    from skeebert.identity import SaltMissing

    with pytest.raises(SaltMissing) as info:
        Pseudonymiser.load(tmp_path / "data" / ".salt", create=False)
    assert str((tmp_path / "data" / ".salt").resolve()) in str(info.value)
    assert not (tmp_path / "data" / ".salt").exists()
