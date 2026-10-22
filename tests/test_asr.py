"""ASR 相关测试：只覆盖不依赖模型下载的部分。"""

from pathlib import Path

import numpy as np
import pytest

from aicut.analysis import asr


def test_load_waveform_decodes_sixteen_k_mono(tmp_path: Path):
    wav = tmp_path / "tone.wav"
    from aicut.analysis.media import ensure_ffmpeg
    import subprocess

    subprocess.run(
        [
            ensure_ffmpeg(), "-hide_banner", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5:sample_rate=44100",
            "-ac", "2", "-ar", "44100", str(wav),
        ],
        check=True,
        capture_output=True,
    )

    audio = asr.load_waveform(wav)
    assert audio.dtype == np.float32
    # 0.5 秒 @16k ≈ 8000 采样
    assert 7_000 < audio.size < 9_000
    assert np.abs(audio).max() > 0.1


def test_load_waveform_reports_missing_file(tmp_path: Path):
    with pytest.raises(RuntimeError):
        asr.load_waveform(tmp_path / "nope.wav")


def test_to_simplified_handles_traditional():
    pytest.importorskip("opencc")
    assert asr._to_simplified("決定講什麼") == "决定讲什么"


def test_to_simplified_falls_back_without_opencc(monkeypatch):
    monkeypatch.setattr(asr, "_OPENCC", False)
    assert asr._to_simplified("決定講什麼") == "決定講什麼"
