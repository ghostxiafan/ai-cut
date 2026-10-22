"""字幕输出：SRT / ASS。

SRT 用于外挂与平台上传，ASS 用于烧制与原样式控制。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class Cue:
    start: float
    end: float
    text: str
    style: str = "caption"


def _srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def write_srt(cues: Iterable[Cue], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for idx, cue in enumerate(cues, start=1):
        text = cue.text.replace("\r\n", "\n").strip()
        lines.append(str(idx))
        lines.append(f"{_srt_time(cue.start)} --> {_srt_time(cue.end)}")
        lines.append(text)
        lines.append("")
    # ffmpeg/libass 需要 UTF-8；BOM 可让部分播放器正确识别中文
    target.write_text("\n".join(lines), encoding="utf-8-sig")
    return target


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Microsoft YaHei,{caption_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,{caption_outline},1,2,60,60,{caption_margin},1
Style: Hook,Microsoft YaHei,{hook_size},&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,2,0,1,{hook_outline},2,5,80,80,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def write_ass(
    cues: Sequence[Cue],
    path: str | Path,
    width: int,
    height: int,
    styles: dict[str, str] | None = None,
) -> Path:
    """把字幕写成 ASS。styles 允许把 IR 里的 style 名映射到 ASS 样式名。"""
    mapping = styles or {}
    scale = min(width, height) / 1080.0
    header = ASS_HEADER.format(
        w=width,
        h=height,
        caption_size=max(18, int(48 * scale)),
        caption_outline=max(1, round(2 * scale)),
        caption_margin=max(40, int(height * 0.08)),
        hook_size=max(20, int(64 * scale)),
        hook_outline=max(1, round(3 * scale)),
    )

    events: list[str] = []
    for cue in cues:
        default = "Hook" if cue.style == "hook" else "Caption"
        style = mapping.get(cue.style, default)
        text = cue.text.replace("\r\n", "\n").replace("\n", r"\N")
        events.append(
            f"Dialogue: 0,{_ass_time(cue.start)},{_ass_time(cue.end)},{style},,0,0,0,,{text}"
        )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(header + "\n".join(events) + "\n", encoding="utf-8-sig")
    return target
