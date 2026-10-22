"""FFmpeg 渲染内核：把 Timeline IR 渲染成 MP4。

设计要点：
- 只用一条 ffmpeg 命令 + filter_complex，避免中间文件与多代编码。
- 视频轨按顺序 concat；静音片段用 anullsrc 补齐，保证 concat 的音频流连续。
- 背景音乐/解说轨用 adelay + amix 混入，避免"重新编码两次"。
- 字幕走 ASS 烧制，中文样式由 subtitle.py 统一控制。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .analysis.media import ensure_ffmpeg
from .ir import Timeline
from .subtitle import Cue, write_ass

SR = 48_000


@dataclass
class RenderResult:
    output: str
    command: list[str]
    duration: float


def _ff_path(path: str | Path) -> str:
    """ffmpeg filter 参数里的路径：正斜杠 + 转义冒号与单引号。"""
    text = str(Path(path).resolve()).replace("\\", "/")
    text = text.replace(":", r"\:").replace("'", r"\'")
    return text


def build_command(
    timeline: Timeline,
    output: str | Path,
    *,
    ass_path: str | Path | None = None,
    crf: int = 20,
    preset: str = "medium",
    threads: int | None = None,
) -> list[str]:
    if not timeline.video:
        raise ValueError("FFmpeg 渲染至少需要一条视频片段")

    ffmpeg = ensure_ffmpeg()
    canvas = timeline.canvas
    cmd: list[str] = [ffmpeg, "-hide_banner", "-y"]

    # 1) 输入：每个视频片段单独 seek，避免全片解码
    for clip in timeline.video:
        if clip.in_:
            cmd += ["-ss", f"{clip.in_:.3f}"]
        cmd += ["-t", f"{clip.duration:.3f}", "-i", clip.src]

    audio_inputs = list(timeline.audio)
    for clip in audio_inputs:
        if clip.loop:
            cmd += ["-stream_loop", "-1"]
        if clip.in_:
            cmd += ["-ss", f"{clip.in_:.3f}"]
        cmd += ["-t", f"{clip.duration:.3f}", "-i", clip.src]

    # 2) 视频轨：统一分辨率/帧率/像素格式后 concat
    filters: list[str] = []
    concat_pads: list[str] = []
    n_video = len(timeline.video)

    for idx, clip in enumerate(timeline.video):
        filters.append(
            f"[{idx}:v]scale={canvas.w}:{canvas.h}:force_original_aspect_ratio=decrease,"
            f"pad={canvas.w}:{canvas.h}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps={canvas.fps},format=yuv420p[v{idx}]"
        )
        if clip.muted:
            filters.append(
                f"anullsrc=channel_layout=stereo:sample_rate={SR},"
                f"atrim=0:{clip.duration:.3f},asetpts=N/SR/TB[a{idx}]"
            )
        else:
            vol = clip.volume
            filters.append(
                f"[{idx}:a]aformat=sample_fmts=fltp:sample_rates={SR}:channel_layouts=stereo,"
                f"volume={vol},atrim=0:{clip.duration:.3f},asetpts=N/SR/TB[a{idx}]"
            )
        concat_pads.append(f"[v{idx}][a{idx}]")

    filters.append(
        "".join(concat_pads) + f"concat=n={n_video}:v=1:a=1[vbase][abase]"
    )

    # 3) 叠加音频轨（BGM/配音），按 start 对齐
    total = timeline.duration
    current_audio = "[abase]"
    for j, clip in enumerate(audio_inputs):
        src_index = n_video + j
        delay_ms = max(0, int(round(clip.start * 1000)))
        filters.append(
            f"[{src_index}:a]aformat=sample_fmts=fltp:sample_rates={SR}:channel_layouts=stereo,"
            f"volume={clip.volume},atrim=0:{min(clip.duration, total):.3f},asetpts=N/SR/TB,"
            f"adelay={delay_ms}|{delay_ms}[a{j}]"
        )
        merged = f"[amix{j}]"
        # normalize=0：避免 amix 自动按输入数衰减，音量完全由每个片段的 volume 决定
        filters.append(
            f"{current_audio}[a{j}]amix=inputs=2:duration=first:dropout_transition=0:normalize=0{merged}"
        )
        current_audio = merged

    if audio_inputs:
        filters.append(f"{current_audio}atrim=0:{total:.3f}[aout]")
        audio_out = "[aout]"
    else:
        audio_out = "[abase]"

    # 4) 烧字幕
    video_out = "[vbase]"
    if ass_path is not None:
        filters.append(f"[vbase]ass='{_ff_path(ass_path)}'[vsub]")
        video_out = "[vsub]"

    cmd += ["-filter_complex", ";".join(filters)]
    cmd += ["-map", video_out, "-map", audio_out]
    cmd += ["-t", f"{total:.3f}"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", str(SR)]
    if threads:
        cmd += ["-threads", str(threads)]
    cmd += ["-movflags", "+faststart", str(Path(output).resolve())]
    return cmd


def render(
    timeline: Timeline,
    output: str | Path,
    *,
    subtitle_path: str | Path | None = None,
    burn_subtitles: bool = True,
    crf: int = 20,
    preset: str = "medium",
    dry_run: bool = False,
    timeout: int = 3600,
) -> RenderResult:
    """渲染时间线。subtitle_path 缺省时用 timeline.text 自动生成 ASS。"""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ass_file: Path | None = None
    if burn_subtitles and timeline.text:
        if subtitle_path is not None:
            ass_file = Path(subtitle_path)
        else:
            ass_file = out_path.with_suffix(".ass")
            cues = [Cue(c.start, c.end, c.text, c.style) for c in timeline.text]
            write_ass(cues, ass_file, timeline.canvas.w, timeline.canvas.h)

    cmd = build_command(timeline, out_path, ass_path=ass_file, crf=crf, preset=preset)
    if dry_run:
        return RenderResult(output=str(out_path), command=cmd, duration=timeline.duration)

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-25:])
        raise RuntimeError(f"ffmpeg 退出码 {proc.returncode}\n命令: {' '.join(cmd)}\n日志:\n{tail}")

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"渲染未产出文件: {out_path}")

    return RenderResult(output=str(out_path), command=cmd, duration=timeline.duration)


def render_with_srt(timeline: Timeline, output: str | Path, srt_path: str | Path, **kwargs) -> RenderResult:
    """使用外部 SRT 烧制（由 subtitle.write_srt 产出或平台下载）。"""
    ass = Path(output).with_suffix(".ass")
    proc = subprocess.run(
        [ensure_ffmpeg(), "-hide_banner", "-y", "-i", str(srt_path), str(ass)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0 or not ass.exists():
        raise RuntimeError(f"SRT 转 ASS 失败: {(proc.stderr or '').strip()[-500:]}")
    return render(timeline, output, subtitle_path=ass, **kwargs)


def describe(timeline: Timeline) -> str:
    """人类可读的时间线摘要，便于 CLI 输出与排查。"""
    lines = [
        f"画布 {timeline.canvas.w}x{timeline.canvas.h} @ {timeline.canvas.fps}fps",
        f"时长 {timeline.duration:.2f}s",
        f"视频 {len(timeline.video)} 段 / 音频 {len(timeline.audio)} 段 / 文字 {len(timeline.text)} 条",
    ]
    for i, c in enumerate(timeline.video):
        lines.append(
            f"  V{i + 1}: {Path(c.src).name} in={c.in_:.2f} dur={c.duration:.2f}"
            f"{' (静音)' if c.muted else ''}"
        )
    for i, c in enumerate(timeline.audio):
        lines.append(f"  A{i + 1}: {Path(c.src).name} start={c.start:.2f} dur={c.duration:.2f} vol={c.volume}")
    return "\n".join(lines)
