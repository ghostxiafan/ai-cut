"""素材理解：转写与切点检测。重量级依赖按需导入，未安装时给出可执行提示。"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Segment:
    start: float
    end: float
    text: str


def transcribe(audio_path: str | Path, model_size: str = "small", language: str = "zh") -> list[Segment]:
    """用 faster-whisper 转写。安装：pip install faster-whisper"""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError("缺少 faster-whisper，请执行：pip install faster-whisper") from exc

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, _info = model.transcribe(str(audio_path), language=language, vad_filter=True)
    return [Segment(round(s.start, 3), round(s.end, 3), s.text.strip()) for s in segments]


def extract_audio(video_path: str | Path, out_path: str | Path) -> Path:
    """抽出 16k 单声道 wav，供 ASR 使用。"""
    from .media import ensure_ffmpeg

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ensure_ffmpeg(), "-hide_banner", "-y",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(target),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"抽音频失败: {(proc.stderr or '')[-400:]}")
    return target


def detect_silence(video_path: str | Path, noise_db: int = -32, min_silence: float = 0.45) -> list[tuple[float, float]]:
    """用 ffmpeg silencedetect 找出静音区间，可据此切成候选片段。"""
    from .media import ensure_ffmpeg

    cmd = [
        ensure_ffmpeg(), "-hide_banner", "-i", str(video_path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    spans: list[tuple[float, float]] = []
    start: float | None = None
    for line in (proc.stderr or "").splitlines():
        if "silence_start:" in line:
            start = float(line.split("silence_start:")[1].strip())
        elif "silence_end:" in line and start is not None:
            end = float(line.split("silence_end:")[1].split("|")[0].strip())
            spans.append((start, end))
            start = None
    return spans
