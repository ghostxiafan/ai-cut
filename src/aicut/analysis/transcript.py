"""转写结果的数据模型与序列化。

支持：
- ASR 直出（analysis.asr.transcribe）-> Transcript
- 读取/写出 SRT、VTT 或纯文本
- 保存/加载 JSON，作为选段与文案的输入

时间单位统一为秒（float），便于与 IR 直接对接。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..subtitle import Cue, write_srt

_SRT_TIME = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)
_CN_PUNCT = "。！？!?…"


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def char_count(self) -> int:
        return len(re.sub(r"\s+", "", self.text))

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"转写片段结束时间({self.end}) 早于开始({self.start})")


@dataclass
class Transcript:
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = "zh"
    source: str = ""

    @property
    def duration(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    def __len__(self) -> int:
        return len(self.segments)

    def plain_text(self, with_timestamps: bool = False) -> str:
        if not with_timestamps:
            return "".join(s.text for s in self.segments)
        return "\n".join(f"[{_mmss(s.start)}] {s.text}" for s in self.segments)

    def slice(self, start: float, end: float) -> list[TranscriptSegment]:
        """取出与 [start, end) 有交集的片段，并按窗口裁剪时间。"""
        out: list[TranscriptSegment] = []
        for seg in self.segments:
            if seg.end <= start or seg.start >= end:
                continue
            out.append(
                TranscriptSegment(
                    max(seg.start, start) - start,
                    min(seg.end, end) - start,
                    seg.text,
                )
            )
        return out

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "source": self.source,
            "duration": round(self.duration, 3),
            "segments": [
                {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
                for s in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        segments = [
            TranscriptSegment(float(s["start"]), float(s["end"]), str(s.get("text", "")))
            for s in data.get("segments", [])
        ]
        return cls(segments=segments, language=data.get("language", "zh"), source=data.get("source", ""))

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "Transcript":
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def to_srt(self, path: str | Path) -> Path:
        cues = [Cue(s.start, s.end, s.text) for s in self.segments]
        return write_srt(cues, path)

    def to_srt_text(self) -> str:
        return "\n".join(
            f"{i}\n{_srt_stamp(s.start)} --> {_srt_stamp(s.end)}\n{s.text}\n"
            for i, s in enumerate(self.segments, start=1)
        )


def _mmss(seconds: float) -> str:
    m, s = divmod(int(max(0.0, seconds)), 60)
    return f"{m:02d}:{s:02d}"


def _srt_stamp(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(text: str) -> Transcript:
    """解析 SRT（兼容 BOM、CRLF、逗号或点号毫秒分隔）。"""
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    segments: list[TranscriptSegment] = []
    for block in re.split(r"\n{2,}", text):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        idx = 0
        if lines[0].strip().isdigit():
            idx = 1
        if idx >= len(lines):
            continue
        match = _SRT_TIME.search(lines[idx])
        if not match:
            continue
        g = [int(x) for x in match.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / (10 ** len(match.group(4)))
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / (10 ** len(match.group(8)))
        body = " ".join(lines[idx + 1 :]).strip()
        if body:
            segments.append(TranscriptSegment(round(start, 3), round(end, 3), body))
    return Transcript(segments=segments)


def load_transcript(path: str | Path) -> Transcript:
    """按扩展名自动选择解析方式：.json / .srt / 其他按纯文本处理。"""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".json":
        return Transcript.load(p)
    if suffix == ".srt":
        return parse_srt(p.read_text(encoding="utf-8", errors="replace"))
    if suffix == ".txt":
        return transcript_from_text(p.read_text(encoding="utf-8", errors="replace"), source=str(p))
    raise ValueError(f"不支持的转写文件格式: {suffix}（支持 .json/.srt/.txt）")


def transcript_from_text(text: str, source: str = "", chars_per_second: float = 5.0) -> Transcript:
    """把无时间戳的纯文本按标点切成片段并估算时间，供快速试跑用。"""
    sentences = [s.strip() for s in re.split(f"(?<=[{_CN_PUNCT}])", text) if s.strip()]
    segments: list[TranscriptSegment] = []
    cursor = 0.0
    for sentence in sentences:
        chars = len(re.sub(r"\s+", "", sentence))
        dur = max(0.8, chars / chars_per_second)
        segments.append(TranscriptSegment(round(cursor, 3), round(cursor + dur, 3), sentence))
        cursor += dur
    return Transcript(segments=segments, source=source)


def segments_from_asr(raw: list) -> Transcript:
    """把 faster-whisper 的 segment 列表转成 Transcript。"""
    return Transcript(
        segments=[TranscriptSegment(round(s.start, 3), round(s.end, 3), s.text.strip()) for s in raw]
    )
