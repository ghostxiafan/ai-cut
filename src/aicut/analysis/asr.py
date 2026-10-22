"""素材理解：转写与切点检测。重量级依赖按需导入，未安装时给出可执行提示。"""

from __future__ import annotations

import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASR_SAMPLE_RATE = 16_000

_OPENCC: object | None = None


def _to_simplified(text: str) -> str:
    """Whisper 在中文下会随机输出繁体，用 OpenCC 统一成简体（未安装则原样返回）。"""
    global _OPENCC
    if _OPENCC is None:
        try:
            import opencc  # type: ignore
        except ImportError:
            _OPENCC = False
            return text
        _OPENCC = opencc.OpenCC("t2s")
    if _OPENCC is False:
        return text
    return _OPENCC.convert(text)


@dataclass
class Segment:
    start: float
    end: float
    text: str


def load_waveform(audio_path: str | Path, sample_rate: int = ASR_SAMPLE_RATE) -> np.ndarray:
    """用 ffmpeg 解成 float32 单声道波形。

    不走 faster-whisper 自带的 PyAV 解码：它依赖 PyAV 的 metadata_errors 参数，
    在新版 PyAV 上已移除，会直接 TypeError。ffmpeg 解码同时兼容更多容器。
    """
    from .media import ensure_ffmpeg

    cmd = [
        ensure_ffmpeg(), "-hide_banner", "-nostdin",
        "-i", str(audio_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace")[-400:]
        raise RuntimeError(f"ffmpeg 解码音频失败: {detail}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def transcribe(
    audio_path: str | Path,
    model_size: str = "medium",
    language: str = "zh",
    *,
    device: str = "auto",
    compute_type: str = "auto",
    vad_filter: bool = True,
    beam_size: int = 5,
    initial_prompt: str | None = "以下是普通话的句子，使用简体中文。",
) -> list[Segment]:
    """用 faster-whisper 转写。安装：pip install faster-whisper

    initial_prompt 让模型倾向输出简体中文：仅指定 language="zh" 时，
    Whisper 常默认输出繁体。

    若素材只有背景音乐、无人声，vad_filter 会把全部音频判为静音并返回空列表，
    这是正常行为；调用方应准备兜底方案（见 highlights.uniform_candidates）。
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError("缺少 faster-whisper，请执行：pip install faster-whisper") from exc

    audio = load_waveform(audio_path)
    if audio.size == 0:
        raise RuntimeError(f"音频为空或无法解码: {audio_path}")

    with warnings.catch_warnings():
        # HF Hub 在 Windows 无开发者模式时会刷符号链接缓存告警，与转写无关
        warnings.filterwarnings("ignore", message=".*symlinks.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, _info = model.transcribe(
        audio,
        language=language,
        vad_filter=vad_filter,
        beam_size=beam_size,
        initial_prompt=initial_prompt,
        condition_on_previous_text=False,
    )
    out: list[Segment] = []
    for seg in segments:
        text = seg.text.strip()
        if language.startswith("zh"):
            text = _to_simplified(text)
        out.append(Segment(round(seg.start, 3), round(seg.end, 3), text))
    return out


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
