from pathlib import Path

import pytest

from aicut import config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path: Path):
    import os

    snapshot = dict(os.environ)
    for var in ("AICUT_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
                "AICUT_LLM_BASE_URL", "AICUT_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    # 隔离真实项目的 .env，避免影响断言
    monkeypatch.setattr(config, "_dotenv_candidates", lambda: [tmp_path / ".env"])
    monkeypatch.setattr(config, "_env_loaded", False, raising=False)
    yield
    # load_env 直接写 os.environ，monkeypatch 感知不到，必须手动还原
    os.environ.clear()
    os.environ.update(snapshot)
    config._env_loaded = False


def test_load_env_reads_dotenv_without_overriding(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text(
        "# 注释\nAICUT_LLM_API_KEY=from-file\nAICUT_LLM_MODEL=deepseek-chat\n空行忽略\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "already-set")

    config.load_env(force=True)
    import os

    assert os.environ["AICUT_LLM_API_KEY"] == "from-file"
    assert os.environ["DEEPSEEK_API_KEY"] == "already-set"


def test_resolve_uses_defaults(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-1234")
    settings = config.resolve()
    assert settings.base_url == "https://api.deepseek.com/v1"
    assert settings.model == "deepseek-flash"
    assert settings.api_key == "sk-test-1234"


def test_resolve_overrides(monkeypatch):
    monkeypatch.setenv("AICUT_LLM_API_KEY", "k")
    settings = config.resolve(base_url="https://example.com/v1/", model="m")
    assert settings.base_url == "https://example.com/v1"
    assert settings.model == "m"


def test_missing_key_raises():
    with pytest.raises(config.ConfigError):
        config.resolve()


@pytest.mark.parametrize(
    "key,expected",
    [("sk-1234567890", "sk-...7890"), ("short", "*****")],
)
def test_masked_key(key: str, expected: str):
    assert config.LLMSettings(key, "u", "m").masked_key == expected


def test_api_key_priority(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek")
    monkeypatch.setenv("AICUT_LLM_API_KEY", "generic")
    assert config.resolve().api_key == "generic"
