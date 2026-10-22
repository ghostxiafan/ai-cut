import subprocess
from pathlib import Path

import pytest

from aicut.analysis.media import ensure_ffmpeg, probe
from aicut.ir import AudioClip, Canvas, TextClip, Timeline, VideoClip
from aicut.render import build_command, render

FFMPEG = ensure_ffmpeg()


def _clip(path: Path, seconds: float, size: str = "320x568") -> Path:
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
            "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("media")
    return _clip(tmp / "a.mp4", 1.0), _clip(tmp / "b.mp4", 1.0)


def test_build_command_has_expected_filters(media):
    a, b = media
    timeline = Timeline(
        canvas=Canvas(w=720, h=1280, fps=30),
        video=[VideoClip(src=str(a), out=1.0, muted=True), VideoClip(src=str(b), out=1.0)],
        audio=[AudioClip(src=str(a), out=2.0, volume=0.2)],
        text=[TextClip(text="钩子", start=0, end=1, style="hook")],
    )
    cmd = build_command(timeline, "out.mp4", ass_path=None)
    joined = " ".join(cmd)
    assert "concat=n=2:v=1:a=1" in joined
    assert "anullsrc" in joined  # 静音片段补轨
    assert "amix=inputs=2" in joined
    assert "libx264" in joined


def test_render_end_to_end(media, tmp_path: Path):
    a, b = media
    timeline = Timeline(
        canvas=Canvas(w=720, h=1280, fps=30),
        video=[VideoClip(src=str(a), out=1.0, muted=True), VideoClip(src=str(b), out=1.0)],
        text=[TextClip(text="测试字幕", start=0.2, end=1.8)],
    )
    out = tmp_path / "out.mp4"
    result = render(timeline, out, preset="ultrafast", crf=30)

    assert Path(result.output).exists()
    info = probe(out)
    assert info.has_video and info.has_audio
    assert (info.width, info.height) == (720, 1280)
    assert abs(info.duration - 2.0) < 0.2
    assert out.with_suffix(".ass").exists()
