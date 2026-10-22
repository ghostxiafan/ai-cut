"""生成自检用素材与时间线，不依赖任何外部素材。"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .analysis.media import ensure_ffmpeg
from .ir import AudioClip, Canvas, TextClip, Timeline, VideoClip

DEMO_TEXT = "3 秒看懂自动剪辑"
DEMO_HOOK = "别再一帧一帧剪了"


def _make_clip(path: Path, pattern: str, seconds: float, freq: int) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ensure_ffmpeg(), "-hide_banner", "-y",
        "-f", "lavfi", "-i", f"{pattern}=size=720x1280:rate=30:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-c:a", "aac", "-shortest",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"生成测试素材失败: {(proc.stderr or '')[-400:]}")
    return path


def _make_bgm(path: Path, seconds: float) -> Path:
    if path.exists():
        return path
    cmd = [
        ensure_ffmpeg(), "-hide_banner", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
        "-c:a", "aac", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"生成 BGM 失败: {(proc.stderr or '')[-400:]}")
    return path


def build_demo(workdir: str | Path) -> Timeline:
    work = Path(workdir).resolve()
    a = _make_clip(work / "clip_a.mp4", "testsrc2", 3.0, 440)
    b = _make_clip(work / "clip_b.mp4", "smptebars", 2.5, 660)
    bgm = _make_bgm(work / "bgm.m4a", 5.5)

    return Timeline(
        canvas=Canvas(w=1080, h=1920, fps=30),
        video=[
            VideoClip(src=str(a), out=3.0, muted=True),
            VideoClip(src=str(b), out=2.5),
        ],
        audio=[AudioClip(src=str(bgm), out=5.5, volume=0.15, loop=True)],
        text=[
            TextClip(text=DEMO_HOOK, start=0.0, end=1.6, style="hook"),
            TextClip(text=DEMO_TEXT, start=1.8, end=4.4, style="caption"),
        ],
    )
