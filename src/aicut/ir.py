"""统一时间线中间表示（IR）。

所有上层分析/文案模块只负责产出 IR，任何出片后端（FFmpeg、剪映草稿、
未来的 PR/达芬奇适配器）只负责消费 IR。换后端不需要改决策逻辑。

IR 的 JSON 形态见 README 示例，也等价于：

    {
      "canvas": {"w": 1080, "h": 1920, "fps": 30},
      "tracks": {
        "video": [{"src": "...", "in": 0, "out": 3.5, "muted": false}],
        "audio": [{"src": "...", "in": 0, "out": 30, "volume": 0.3, "loop": true}],
        "text":  [{"text": "...", "start": 0, "end": 2.8, "style": "hook"}]
      }
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class IRValidationError(ValueError):
    """IR 结构不合法。"""


@dataclass
class Canvas:
    w: int = 1080
    h: int = 1920
    fps: int = 30

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise IRValidationError(f"画布尺寸非法: {self.w}x{self.h}")
        if self.fps <= 0:
            raise IRValidationError(f"帧率非法: {self.fps}")

    @property
    def aspect(self) -> float:
        return self.w / self.h


@dataclass
class VideoClip:
    src: str
    out: float
    start: float | None = None  # 时间线绝对起点；缺省则顺序排列
    in_: float = 0.0
    muted: bool = False
    volume: float = 1.0

    @property
    def duration(self) -> float:
        return self.out - self.in_

    def __post_init__(self) -> None:
        if self.out <= self.in_:
            raise IRValidationError(f"{self.src}: out({self.out}) 必须大于 in({self.in_})")


@dataclass
class AudioClip:
    src: str
    out: float
    start: float = 0.0
    in_: float = 0.0
    volume: float = 1.0
    loop: bool = False

    @property
    def duration(self) -> float:
        return self.out - self.in_


@dataclass
class TextClip:
    text: str
    start: float
    end: float
    style: str = "caption"

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise IRValidationError(f"文字片段结束时间({self.end}) 必须晚于开始时间({self.start})")


@dataclass
class Timeline:
    canvas: Canvas = field(default_factory=Canvas)
    video: list[VideoClip] = field(default_factory=list)
    audio: list[AudioClip] = field(default_factory=list)
    text: list[TextClip] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """成片时长：以视频轨为准；没有视频轨时退回音频/文字最晚结束点。"""
        if self.video:
            return sum(c.duration for c in self.video)
        ends = [c.out for c in self.audio] + [c.end for c in self.text]
        return max(ends, default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canvas": asdict(self.canvas),
            "tracks": {
                "video": [asdict(c) for c in self.video],
                "audio": [asdict(c) for c in self.audio],
                "text": [asdict(c) for c in self.text],
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _num(raw: Any, key: str, default: float | None = None) -> float:
    if raw is None:
        if default is None:
            raise IRValidationError(f"缺少数值字段: {key}")
        return default
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise IRValidationError(f"字段 {key} 不是数值: {raw!r}") from exc


def _resolve(base_dir: Path | None, src: str) -> str:
    """相对路径优先按 IR 文件所在目录解析，其次退回当前工作目录。"""
    path = Path(src)
    if path.is_absolute():
        return str(path)
    if base_dir is not None:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return str(candidate)
    if path.exists():
        return str(path.resolve())
    return str((base_dir / path).resolve()) if base_dir is not None else str(path)


def timeline_from_dict(data: dict[str, Any], base_dir: str | Path | None = None) -> Timeline:
    if not isinstance(data, dict):
        raise IRValidationError("IR 根节点必须是对象")

    base = Path(base_dir) if base_dir is not None else None
    canvas_raw = data.get("canvas") or {}
    canvas = Canvas(
        w=int(_num(canvas_raw.get("w"), "canvas.w", 1080)),
        h=int(_num(canvas_raw.get("h"), "canvas.h", 1920)),
        fps=int(_num(canvas_raw.get("fps"), "canvas.fps", 30)),
    )

    tracks = data.get("tracks")
    if tracks is None:
        # 允许扁平写法：直接在根节点给 video/audio/text
        tracks = {k: data.get(k, []) for k in ("video", "audio", "text")}
    if not isinstance(tracks, dict):
        raise IRValidationError("tracks 必须是对象")

    videos: list[VideoClip] = []
    for idx, item in enumerate(tracks.get("video") or []):
        if not item.get("src"):
            raise IRValidationError(f"video[{idx}] 缺少 src")
        out = _num(item.get("out"), f"video[{idx}].out")
        start = item.get("start")
        videos.append(
            VideoClip(
                src=_resolve(base, item["src"]),
                out=out,
                start=None if start is None else _num(start, f"video[{idx}].start"),
                in_=_num(item.get("in"), f"video[{idx}].in", 0.0),
                muted=bool(item.get("muted", False)),
                volume=_num(item.get("volume"), f"video[{idx}].volume", 1.0),
            )
        )

    audios: list[AudioClip] = []
    for idx, item in enumerate(tracks.get("audio") or []):
        if not item.get("src"):
            raise IRValidationError(f"audio[{idx}] 缺少 src")
        audios.append(
            AudioClip(
                src=_resolve(base, item["src"]),
                out=_num(item.get("out"), f"audio[{idx}].out"),
                start=_num(item.get("start"), f"audio[{idx}].start", 0.0),
                in_=_num(item.get("in"), f"audio[{idx}].in", 0.0),
                volume=_num(item.get("volume"), f"audio[{idx}].volume", 1.0),
                loop=bool(item.get("loop", False)),
            )
        )

    texts: list[TextClip] = []
    for idx, item in enumerate(tracks.get("text") or []):
        if not item.get("text"):
            raise IRValidationError(f"text[{idx}] 缺少 text")
        texts.append(
            TextClip(
                text=str(item["text"]),
                start=_num(item.get("start"), f"text[{idx}].start", 0.0),
                end=_num(item.get("end"), f"text[{idx}].end"),
                style=str(item.get("style", "caption")),
            )
        )

    if not videos and not audios:
        raise IRValidationError("时间线为空：至少需要一条视频或音频")

    return Timeline(canvas=canvas, video=videos, audio=audios, text=texts)


def load_timeline(path: str | Path) -> Timeline:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return timeline_from_dict(data, base_dir=p.parent)


def save_timeline(timeline: Timeline, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(timeline.to_json(), encoding="utf-8")
    return p
