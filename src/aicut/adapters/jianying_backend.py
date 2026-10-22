"""剪映草稿后端（预留）。

现状（实测剪映专业版 7.x）：
- 草稿目录下的 draft_content.json / draft_meta_info.json 均为加密内容，
  解码后无 JSON 结构，无法用 pyJianYingDraft 之流的明文草稿方案。
- 草稿根目录可能被用户改到其他盘，实际路径记录在 root_meta_info.json 中，
  本模块会据此自动解析。

可用的解锁条件（任一满足即可启用本后端）：
1. 升级到剪映 11.5+，配合 jy-draftc 调用 videoeditor.dll 完成加解密与无头导出；
2. 换用仍写明文草稿的旧版本客户端，走 pyJianYingDraft。

启用时只需实现 render()：把 Timeline 转成剪映草稿目录（或明文 JSON），
再可选调用 jy-draftc 解密/回写/导出。上层代码无需改动。

如需指定草稿目录，可设置环境变量 AICUT_JIANYING_DRAFT_ROOT。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..ir import Timeline
from .base import BackendError


def _draft_root_hints() -> tuple[Path, ...]:
    """默认草稿目录 + 可选的用户覆盖，不硬编码任何人的具体路径。"""
    hints: list[Path] = []
    override = os.environ.get("AICUT_JIANYING_DRAFT_ROOT")
    if override:
        hints.append(Path(override))
    hints.append(
        Path(os.path.expandvars(r"%LOCALAPPDATA%\JianyingPro\User Data\Projects\com.lveditor.draft"))
    )
    return tuple(hints)


def _root_from_meta(hint: Path) -> Path | None:
    """剪映允许把草稿根目录改到别的盘，真实路径记录在 root_meta_info.json。"""
    meta = hint / "root_meta_info.json"
    if not meta.is_file():
        return None
    try:
        data = json.loads(meta.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None

    candidates: list[Path] = []
    root = data.get("root_path")
    if root:
        candidates.append(Path(root))
    for item in data.get("all_draft_store") or []:
        for key in ("draft_root_path", "draft_fold_path"):
            value = item.get(key)
            if not value:
                continue
            path = Path(value)
            candidates.append(path.parent if key == "draft_fold_path" else path)

    for path in candidates:
        if not path.exists():
            continue
        if any((d / "draft_content.json").is_file() for d in path.iterdir() if d.is_dir()):
            return path
    return candidates[0] if candidates and candidates[0].exists() else None


def _drafts_are_encrypted(root: Path) -> bool | None:
    """抽样检查草稿是否为密文；无法判断时返回 None。"""
    if not root.exists():
        return None
    for draft in root.iterdir():
        content = draft / "draft_content.json"
        if not content.is_file():
            continue
        try:
            head = content.read_bytes()[:1]
        except OSError:
            continue
        return head != b"{"
    return None


class JianyingBackend:
    name = "jianying"

    def __init__(self, draft_root: str | Path | None = None) -> None:
        self.draft_root = Path(draft_root) if draft_root else None

    def _resolve_root(self) -> Path | None:
        if self.draft_root is not None:
            if not self.draft_root.exists():
                return None
            # 显式指定时同样支持"入口目录"：真实路径可能记在 root_meta_info.json 里
            real = _root_from_meta(self.draft_root)
            if real and real.exists():
                return real
            return self.draft_root
        for hint in _draft_root_hints():
            if not hint.exists():
                continue
            real = _root_from_meta(hint)
            if real and real.exists():
                return real
            if any((d / "draft_content.json").is_file() for d in hint.iterdir() if d.is_dir()):
                return hint
        return None

    def available(self) -> bool:
        root = self._resolve_root()
        if root is None:
            return False
        return _drafts_are_encrypted(root) is False

    def diagnose(self) -> dict:
        root = self._resolve_root()
        encrypted = _drafts_are_encrypted(root) if root else None
        return {
            "backend": self.name,
            "draft_root": str(root) if root else None,
            "encrypted": encrypted,
            "available": encrypted is False,
        }

    def build_draft(self, timeline: Timeline, name: str) -> Path:
        """仅写出 IR 快照，供具备解密能力的工具二次加工。"""
        raise BackendError(self._explain())

    def render(self, timeline: Timeline, output: str, **kwargs) -> str:
        raise BackendError(self._explain())

    def _explain(self) -> str:
        info = self.diagnose()
        if info["draft_root"] is None:
            return "未找到剪映草稿目录，无法使用剪映后端"
        return (
            f"剪映后端当前不可用：草稿目录 {info['draft_root']} 的 draft_content.json 为加密格式。\n"
            "解决路径：升级剪映至 11.5+ 并接入 jy-draftc 的 videoeditor.dll 加解密，"
            "或改用明文草稿的旧版客户端。\n"
            "在此之前请使用 --backend ffmpeg 出片。"
        )


def dump_timeline_snapshot(timeline: Timeline, path: str | Path) -> Path:
    """把 IR 写成 JSON，作为交给外部工具的中间产物。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(timeline.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return target
