"""剪辑计划：把选中的候选片段 + 转写，转成可渲染的 Timeline IR。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .analysis.highlights import Candidate
from .analysis.transcript import Transcript, TranscriptSegment
from .ir import AudioClip, Canvas, TextClip, Timeline, VideoClip


@dataclass
class PlanOptions:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    subtitles: bool = True
    muzzle_source: bool = False  # True 时压低/静音原声，仅保留字幕与 BGM
    hook: str | None = None
    auto_hook: bool = True
    hook_duration: float = 2.6
    bgm: str | None = None
    bgm_volume: float = 0.12
    max_chars_per_cue: int = 24
    title: str = ""


def _split_long(seg: TranscriptSegment, limit: int) -> list[TranscriptSegment]:
    """过长的字幕按标点二次切分，无标点可用时按字数硬切。"""
    text = seg.text.strip()
    if len(re.sub(r"\s+", "", text)) <= limit:
        return [seg]

    pieces: list[str] = []
    for part in re.split(r"(?<=[，。！？、；!?,;])", text):
        clean = re.sub(r"\s+", "", part)
        if not clean:
            continue
        if len(clean) <= limit:
            pieces.append(part.strip())
        else:
            pieces.extend(clean[i : i + limit] for i in range(0, len(clean), limit))

    if len(pieces) <= 1:
        return [seg]

    total = sum(len(re.sub(r"\s+", "", p)) for p in pieces) or 1
    out: list[TranscriptSegment] = []
    cursor = seg.start
    for i, part in enumerate(pieces):
        if i == len(pieces) - 1:
            end = seg.end
        else:
            end = cursor + seg.duration * len(re.sub(r"\s+", "", part)) / total
        out.append(TranscriptSegment(round(cursor, 3), round(end, 3), part.strip()))
        cursor = end
    return out


def _auto_hook(candidate: Candidate) -> str:
    """从候选开头抽一句短钩子，超长则截断到标点。"""
    text = re.sub(r"\s+", "", candidate.text)
    sentence = re.split(r"(?<=[。！？!?])", text)[0]
    sentence = sentence.strip("，。！？!?,;、")
    if len(sentence) <= 16:
        return sentence
    return sentence[:16]


def _norm(text: str) -> str:
    return re.sub(r"[\s，。！？!?,;、…·\-—\"'“”‘’]", "", text)


def _is_duplicate_hook(caption: str, hook: str) -> bool:
    """钩子与首句字幕内容重复时只保留钩子，避免同一句话叠两层。"""
    a, b = _norm(caption), _norm(hook)
    if not a or not b:
        return False
    return a in b or b in a


def build_timeline(
    media: str | Path,
    transcript: Transcript,
    candidates: list[Candidate],
    options: PlanOptions | None = None,
) -> Timeline:
    if not candidates:
        raise ValueError("没有可用的候选片段，请先运行选段（aicut select）")

    opt = options or PlanOptions()
    media_path = str(Path(media).resolve())

    videos: list[VideoClip] = []
    texts: list[TextClip] = []
    offset = 0.0
    hook_text = ""
    hook_end = 0.0

    for idx, cand in enumerate(candidates):
        videos.append(
            VideoClip(
                src=media_path,
                in_=round(cand.start, 3),
                out=round(cand.end, 3),
                muted=opt.muzzle_source,
            )
        )

        if idx == 0:
            hook_text = opt.hook if opt.hook is not None else (_auto_hook(cand) if opt.auto_hook else "")
            hook_text = hook_text.strip()
            if hook_text:
                hook_end = round(min(opt.hook_duration, cand.duration), 3)
                texts.append(TextClip(text=hook_text, start=0.0, end=hook_end, style="hook"))

        if opt.subtitles:
            for seg in transcript.slice(cand.start, cand.end):
                for piece in _split_long(seg, opt.max_chars_per_cue):
                    if idx == 0 and hook_text and piece.start < hook_end and _is_duplicate_hook(piece.text, hook_text):
                        continue
                    texts.append(
                        TextClip(
                            text=piece.text,
                            start=round(offset + piece.start, 3),
                            end=round(offset + piece.end, 3),
                            style="caption",
                        )
                    )

        offset += cand.duration

    total = offset
    audios: list[AudioClip] = []
    if opt.bgm:
        audios.append(
            AudioClip(
                src=str(Path(opt.bgm).resolve()),
                out=round(total, 3),
                volume=opt.bgm_volume,
                loop=True,
            )
        )

    return Timeline(
        canvas=Canvas(w=opt.width, h=opt.height, fps=opt.fps),
        video=videos,
        audio=audios,
        text=texts,
    )


def plan_summary(candidates: list[Candidate], timeline: Timeline) -> dict:
    return {
        "clips": [
            {
                "index": i + 1,
                "start": round(c.start, 2),
                "end": round(c.end, 2),
                "duration": round(c.duration, 2),
                "score": round(c.score, 1),
                "reasons": {k: round(v, 1) for k, v in c.reasons.items()},
                "text": c.text[:120],
            }
            for i, c in enumerate(candidates)
        ],
        "timeline": {
            "duration": round(timeline.duration, 2),
            "canvas": f"{timeline.canvas.w}x{timeline.canvas.h}@{timeline.canvas.fps}",
            "video_clips": len(timeline.video),
            "audio_clips": len(timeline.audio),
            "text_clips": len(timeline.text),
        },
    }
