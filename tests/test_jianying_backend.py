import json
from pathlib import Path

from aicut.adapters import jianying_backend as jb


def test_hints_include_env_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AICUT_JIANYING_DRAFT_ROOT", str(tmp_path))
    hints = jb._draft_root_hints()
    assert hints[0] == tmp_path
    assert all(isinstance(h, Path) for h in hints)


def test_hints_default_without_override(monkeypatch):
    monkeypatch.delenv("AICUT_JIANYING_DRAFT_ROOT", raising=False)
    hints = jb._draft_root_hints()
    assert hints
    assert "JianyingPro" in str(hints[0])


def test_resolve_root_follows_meta(tmp_path: Path, monkeypatch):
    """草稿目录被改到别的盘时，应能通过 root_meta_info.json 找到真实路径。"""
    real = tmp_path / "elsewhere"
    draft = real / "我的项目"
    draft.mkdir(parents=True)
    (draft / "draft_content.json").write_text('{"tracks": {}}', encoding="utf-8")

    hint = tmp_path / "hint"
    hint.mkdir()
    (hint / "root_meta_info.json").write_text(
        json.dumps({"root_path": str(real).replace("\\", "/")}, ensure_ascii=False),
        encoding="utf-8",
    )

    backend = jb.JianyingBackend(draft_root=hint)
    assert backend._resolve_root() == real


def test_encrypted_detection(tmp_path: Path):
    root = tmp_path / "drafts"
    plain = root / "plain"
    plain.mkdir(parents=True)
    (plain / "draft_content.json").write_text('{"tracks": {}}', encoding="utf-8")
    assert jb._drafts_are_encrypted(root) is False

    (plain / "draft_content.json").write_text("bBB1cU78GfBVe8m4", encoding="utf-8")
    assert jb._drafts_are_encrypted(root) is True

    assert jb._drafts_are_encrypted(tmp_path / "missing") is None


def test_unavailable_backend_explains(tmp_path: Path):
    root = tmp_path / "drafts"
    draft = root / "a"
    draft.mkdir(parents=True)
    (draft / "draft_content.json").write_text("ENCRYPTED", encoding="utf-8")

    backend = jb.JianyingBackend(draft_root=root)
    info = backend.diagnose()
    assert info["encrypted"] is True
    assert info["available"] is False
    assert "11.5" in backend._explain()


def test_available_when_plaintext(tmp_path: Path):
    root = tmp_path / "drafts"
    draft = root / "a"
    draft.mkdir(parents=True)
    (draft / "draft_content.json").write_text('{"tracks": {}}', encoding="utf-8")

    backend = jb.JianyingBackend(draft_root=root)
    assert backend.available() is True
