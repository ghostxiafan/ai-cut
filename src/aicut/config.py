"""LLM 配置解析。

密钥只从环境变量或 .env 读取，绝不写入源码或提交到仓库。
.env 已在 .gitignore 中排除。

支持的变量（优先级从高到低）：
    AICUT_LLM_API_KEY  -> 通用
    DEEPSEEK_API_KEY   -> DeepSeek 专用
    OPENAI_API_KEY     -> 兜底

    AICUT_LLM_BASE_URL / AICUT_LLM_MODEL 可覆盖默认值。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

_API_KEY_VARS = ("AICUT_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY")

_env_loaded = False


class ConfigError(RuntimeError):
    """缺少必要的配置。"""


def _dotenv_candidates() -> list[Path]:
    """依次查找：当前目录、向上最近的含 pyproject.toml 的项目根目录。"""
    candidates: list[Path] = [Path.cwd() / ".env"]
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            candidates.append(parent / ".env")
            break
    return candidates


def load_env(force: bool = False) -> None:
    """把 .env 里的键值载入 os.environ（不覆盖已存在的变量）。"""
    global _env_loaded
    if _env_loaded and not force:
        return
    for candidate in _dotenv_candidates():
        if not candidate.is_file():
            continue
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    _env_loaded = True


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str
    model: str

    @property
    def masked_key(self) -> str:
        """日志与报错里只显示前缀与末四位。"""
        key = self.api_key
        if len(key) <= 8:
            return "*" * len(key)
        return f"{key[:3]}...{key[-4:]}"


def resolve(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> LLMSettings:
    load_env()

    key = api_key or next((os.environ[v] for v in _API_KEY_VARS if os.environ.get(v)), None)
    if not key:
        raise ConfigError(
            "缺少 API Key：在项目根目录新建 .env 写入 AICUT_LLM_API_KEY=你的密钥，"
            "或设置同名环境变量。"
        )
    return LLMSettings(
        api_key=key,
        base_url=(base_url or os.environ.get("AICUT_LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
        model=model or os.environ.get("AICUT_LLM_MODEL") or DEFAULT_MODEL,
    )
