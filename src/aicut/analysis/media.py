"""媒体探针：只依赖 ffprobe，用于素材校验与时长计算。"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FFmpegMissingError(RuntimeError):
    """PATH 中找不到 ffmpeg/ffprobe。"""


@dataclass
class MediaInfo:
    path: str
    duration: float
    width: int
    height: int
    has_audio: bool
    has_video: bool


def _bin(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise FFmpegMissingError(
            f"未找到 {name}，请安装 FFmpeg 并加入 PATH（Windows: winget install Gyan.FFmpeg）"
        )
    return exe


def probe(path: str | Path) -> MediaInfo:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"素材不存在: {target}")

    cmd = [
        _bin("ffprobe"),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(target),
    ]
    raw = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True)
    data = json.loads(raw.stdout or "{}")

    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = 0.0
    fmt_dur = (data.get("format") or {}).get("duration")
    if fmt_dur:
        duration = float(fmt_dur)
    elif video and video.get("duration"):
        duration = float(video["duration"])

    return MediaInfo(
        path=str(target),
        duration=duration,
        width=int(video.get("width", 0)) if video else 0,
        height=int(video.get("height", 0)) if video else 0,
        has_audio=audio is not None,
        has_video=video is not None,
    )


def ensure_ffmpeg() -> str:
    return _bin("ffmpeg")
