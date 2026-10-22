"""候选片段生成与打分。

默认使用纯本地启发式（不需要任何 API），把长视频切成若干"可成片"的窗口，
再按钩子强度、信息密度、句子完整性、时长贴合度打分。也支持接 LLM 复排。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .transcript import Transcript, TranscriptSegment

# 高唤醒词：制造反差、悬念、利益点，短视频开场最有效
HOOK_WORDS = (
    "其实", "千万别", "不要再", "别再", "很多人", "大多数人", "所有人",
    "真相", "秘密", "为什么", "怎么", "如何", "记住", "第一", "最后",
    "免费", "亏", "赚", "省钱", "教你", "核心", "关键", "重点",
    "没想到", "竟然", "居然", "后悔", "注意", "警告", "坑",
)
# 收束词：句末有结论感，切在这里完播率更高
CLOSER_WORDS = ("所以", "因此", "总之", "记住", "这就是", "关键是", "一句话", "最后")
STRONG_WORDS = ("一定", "必须", "千万", "绝对", "唯一", "最重要", "最", "第一")

_IDEAL_DENSITY = (3.0, 7.5)  # 汉字/秒，中文口播的舒适区


@dataclass
class Candidate:
    start: float
    end: float
    text: str
    score: float = 0.0
    reasons: dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "score": round(self.score, 1),
            "reasons": {k: round(v, 1) for k, v in self.reasons.items()},
            "text": self.text,
        }


def _clean(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _hook_component(text: str) -> float:
    head = _clean(text)[:30]
    score = 0.0
    if any(w in head for w in HOOK_WORDS):
        score += 14.0
    if any(w in head for w in STRONG_WORDS):
        score += 8.0
    if re.search(r"[？?]", text[:60]):
        score += 10.0
    if re.search(r"\d", head):
        score += 7.0
    return min(score, 30.0)


def _density_component(chars: int, duration: float) -> float:
    if duration <= 0:
        return 0.0
    density = chars / duration
    low, high = _IDEAL_DENSITY
    if low <= density <= high:
        return 20.0
    if density < low:
        return max(0.0, 20.0 * (density / low))
    return max(0.0, 20.0 - (density - high) * 4.0)


def _completeness_component(text: str) -> float:
    tail = _clean(text)[-12:]
    if not tail:
        return 0.0
    if re.search(r"[。！？!?…]", text[-3:]):
        return 15.0
    if any(w in tail for w in CLOSER_WORDS):
        return 12.0
    return 4.0


def _length_component(duration: float, target: float) -> float:
    # 目标时长附近给满分，偏离越多衰减越快
    ratio = duration / target if target > 0 else 1.0
    penalty = abs(ratio - 1.0) * 45.0
    return max(0.0, 20.0 - penalty)


def _position_component(start: float) -> float:
    # 略偏好前 1/3 的内容：通常是提问/抛钩子的段落
    if start < 2.0:
        return 8.0
    if start < 60.0:
        return 15.0
    return 12.0


def score_candidate(candidate: Candidate, transcript: Transcript, target: float) -> Candidate:
    text = candidate.text
    chars = len(_clean(text))
    candidate.reasons = {
        "钩子": _hook_component(text),
        "密度": _density_component(chars, candidate.duration),
        "完整性": _completeness_component(text),
        "时长": _length_component(candidate.duration, target),
        "位置": _position_component(candidate.start),
    }
    candidate.score = min(100.0, sum(candidate.reasons.values()))
    return candidate


def build_candidates(
    transcript: Transcript,
    *,
    target: float = 30.0,
    min_duration: float = 12.0,
    max_duration: float = 60.0,
) -> list[Candidate]:
    """在句子边界上滑窗：每个起点尽量延伸到目标时长，产出候选窗口。"""
    segs = [s for s in transcript.segments if s.text.strip()]
    if not segs:
        return []

    candidates: list[Candidate] = []
    n = len(segs)
    for i in range(n):
        start = segs[i].start
        text_parts: list[str] = []
        j = i
        while j < n:
            seg = segs[j]
            parts = text_parts + [seg.text]
            duration = seg.end - start
            if duration > max_duration:
                break
            text_parts = parts
            j += 1
            if duration >= min_duration:
                candidates.append(
                    score_candidate(
                        Candidate(start=start, end=seg.end, text="".join(text_parts)),
                        transcript,
                        target,
                    )
                )
                # 达到目标时长后停止延伸，避免同一内容产生过多近重复窗口
                if duration >= target:
                    break
    return candidates


def rank_candidates(
    candidates: list[Candidate],
    *,
    top: int = 1,
    min_gap: float = 3.0,
) -> list[Candidate]:
    """按分数取前 N 个，并保证彼此不重叠（间隔至少 min_gap 秒）。"""
    picked: list[Candidate] = []
    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        if any(not (cand.end + min_gap <= p.start or cand.start >= p.end + min_gap) for p in picked):
            continue
        picked.append(cand)
        if len(picked) >= top:
            break
    return sorted(picked, key=lambda c: c.start)


def select(
    transcript: Transcript,
    *,
    target: float = 30.0,
    max_duration: float = 60.0,
    top: int = 1,
) -> list[Candidate]:
    return rank_candidates(
        build_candidates(transcript, target=target, max_duration=max_duration),
        top=top,
    )


def subtitle_cues_for(candidate: Candidate, transcript: Transcript) -> list[TranscriptSegment]:
    """取候选窗口内的转写片段，时间平移到候选起点，供字幕使用。"""
    return transcript.slice(candidate.start, candidate.end)


LLM_RANK_PROMPT = """你是短视频编导。下面是从同一支视频里切出的候选片段，请挑出最适合单独成片的一段。
评判标准：开场 3 秒能否抓住注意力、信息是否完整、结尾是否有力、是否需要上下文才能看懂。
只输出 JSON：{"index": 序号, "reason": "20 字以内理由"}"""


def rerank_with_llm(
    candidates: list[Candidate],
    *,
    top: int = 1,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> list[Candidate]:
    """用 LLM 对启发式 Top-K 做复排；失败时原样返回，不影响主流程。"""
    import os
    import urllib.error
    import urllib.request

    if not candidates:
        return []
    # 只送前 8 个，控制成本和延迟
    pool = sorted(candidates, key=lambda c: c.score, reverse=True)[:8]
    listing = [
        {"index": i, "duration": round(c.duration), "text": c.text[:400]} for i, c in enumerate(pool)
    ]

    key = api_key or os.environ.get("AICUT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        return pool[:top]
    base = (base_url or os.environ.get("AICUT_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    mdl = model or os.environ.get("AICUT_LLM_MODEL") or "gpt-4o-mini"

    payload = {
        "model": mdl,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": LLM_RANK_PROMPT},
            {"role": "user", "content": json.dumps(listing, ensure_ascii=False)},
        ],
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"].strip()
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        index = int(json.loads(content)["index"])
        if 0 <= index < len(pool):
            chosen = pool[index]
            rest = [c for c in pool if c is not chosen]
            return [chosen, *rest][:top]
    except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError):
        pass
    return pool[:top]
