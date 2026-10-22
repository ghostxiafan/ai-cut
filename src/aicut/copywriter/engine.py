"""文案引擎：根据转写文本生成钩子、标题、口播稿与标签。

只依赖标准库（urllib），兼容任意 OpenAI 兼容的 /chat/completions 接口，
因此 DeepSeek / Qwen / 本地 vLLM 都能直接用。密钥由 config.resolve 统一解析。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from ..config import ConfigError, resolve

SYSTEM_PROMPT = """你是短视频爆款文案编导。基于给出的视频转写内容，产出中文文案。
要求：
1. hook：3 秒内抓住注意力的开场句，不超过 18 字，可用反差/悬念/利益点。
2. title：不超过 20 字，适合平台标题位，不要标题党到失真。
3. oral_script：可直接配音的口播稿，紧凑、口语化、有节奏。
4. chapters：按内容切分的小节标题。
5. hashtags：5-8 个，含 1-2 个大词与 2-3 个垂类长尾词。
6. cover_text：封面大字，不超过 8 字。
7. comment_bait：一句引导评论的话。
只输出 JSON，不要任何解释或 Markdown 代码块。"""

USER_TEMPLATE = """视频主题：{topic}
目标平台：{platform}
转写文本（可能含时间戳）：
{transcript}"""


@dataclass
class CopyResult:
    hook: str = ""
    title: str = ""
    oral_script: str = ""
    chapters: list[str] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)
    cover_text: str = ""
    comment_bait: str = ""
    raw: str = ""

    def to_dict(self) -> dict:
        return {
            "hook": self.hook,
            "title": self.title,
            "oral_script": self.oral_script,
            "chapters": self.chapters,
            "hashtags": self.hashtags,
            "cover_text": self.cover_text,
            "comment_bait": self.comment_bait,
        }


class LLMError(RuntimeError):
    pass


def _post(url: str, payload: dict, api_key: str, timeout: int = 180) -> dict:
    headers = {"Content-Type": "application/json", "Authorization": " ".join(("Bearer", api_key))}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise LLMError(f"模型接口返回 {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"无法连接模型接口 {url}: {exc.reason}") from exc


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _normalize_hashtags(tags: list[str], limit: int = 8) -> list[str]:
    """统一成带 # 的形式并去重：模型时常漏写 # 或大小写不一。"""
    seen: set[str] = set()
    out: list[str] = []
    for raw in tags:
        tag = raw.strip().lstrip("#").strip()
        # 标签内不允许空格，否则平台会截断
        tag = tag.replace(" ", "").replace("　", "")
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(f"#{tag}")
        if len(out) >= limit:
            break
    return out


def _parse(raw: str) -> CopyResult:
    data = json.loads(_strip_code_fence(raw))
    return CopyResult(
        hook=str(data.get("hook", "")).strip(),
        title=str(data.get("title", "")).strip(),
        oral_script=str(data.get("oral_script", "")).strip(),
        chapters=[str(x).strip() for x in data.get("chapters", []) or [] if str(x).strip()],
        hashtags=_normalize_hashtags([str(x) for x in data.get("hashtags", []) or []]),
        cover_text=str(data.get("cover_text", "")).strip(),
        comment_bait=str(data.get("comment_bait", "")).strip(),
        raw=raw,
    )


def generate(
    transcript: str,
    *,
    topic: str = "",
    platform: str = "抖音",
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.8,
) -> CopyResult:
    settings = resolve(api_key, base_url, model)

    payload = {
        "model": settings.model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    topic=topic or "未指定",
                    platform=platform,
                    transcript=transcript[:12000],
                ),
            },
        ],
    }
    data = _post(f"{settings.base_url}/chat/completions", payload, settings.api_key)
    try:
        raw = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"响应结构异常: {json.dumps(data)[:400]}") from exc
    return _parse(raw)


# ---------------------------------------------------------------- 多候选 + 自评分

CRITIC_PROMPT = """你是短视频内容审核与运营专家。给你若干条候选文案，请逐条打分。

评分维度（各 0-20 分，总分 100）：
- hook：3 秒内抓住注意力的强度，是否有反差/悬念/利益点
- emotion：情绪张力，能否让人想看完、想评论
- clarity：信息密度与可懂度，不绕、不空
- platform：与目标平台的匹配度（标题长度、标签风格、合规）
- finish：完播预期，开头是否直接、结尾是否有收束

要求：
1. 严格打分，不要给所有候选相同的分，必须区分出高低。
2. 每条给出不超过 20 字的关键问题或亮点。
3. 只输出 JSON 数组，不要 Markdown 代码块、不要解释。
格式：[{"index": 0, "hook": 18, "emotion": 16, "clarity": 15, "platform": 17, "finish": 14, "note": "钩子有反差但标题偏虚"}]"""


@dataclass
class ScoredCandidate:
    result: CopyResult
    scores: dict[str, float] = field(default_factory=dict)
    note: str = ""

    @property
    def total(self) -> float:
        return sum(self.scores.values())

    def to_dict(self) -> dict:
        return {
            "total": round(self.total, 1),
            "scores": {k: round(v, 1) for k, v in self.scores.items()},
            "note": self.note,
            **self.result.to_dict(),
        }


_SCORE_KEYS = ("hook", "emotion", "clarity", "platform", "finish")


def _extract_json_list(raw: str) -> list[dict]:
    text = _strip_code_fence(raw)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise LLMError(f"评分结果不是 JSON 数组: {raw[:200]}")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        raise LLMError("评分结果应为数组")
    return [d for d in data if isinstance(d, dict)]


def _score_candidates(
    candidates: list[CopyResult],
    *,
    platform: str,
    settings,
    temperature: float = 0.2,
) -> list[ScoredCandidate]:
    listing = [
        {"index": i, "hook": c.hook, "title": c.title, "cover_text": c.cover_text,
         "hashtags": c.hashtags, "oral_script": c.oral_script[:300]}
        for i, c in enumerate(candidates)
    ]
    payload = {
        "model": settings.model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user", "content": f"目标平台：{platform}\n\n候选文案：\n" + json.dumps(listing, ensure_ascii=False)},
        ],
    }
    data = _post(f"{settings.base_url}/chat/completions", payload, settings.api_key)
    try:
        raw = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"评分响应结构异常: {json.dumps(data)[:300]}") from exc

    scored: list[ScoredCandidate] = []
    for item in _extract_json_list(raw):
        idx = item.get("index")
        if not isinstance(idx, int) or not 0 <= idx < len(candidates):
            continue
        count = sum(1 for s in scored if s.result is candidates[idx])
        if count:
            continue
        scores = {}
        for key in _SCORE_KEYS:
            try:
                scores[key] = max(0.0, min(20.0, float(item.get(key, 0))))
            except (TypeError, ValueError):
                scores[key] = 0.0
        scored.append(ScoredCandidate(candidates[idx], scores, str(item.get("note", ""))))

    # 模型漏掉的候选按 0 分补齐，保证返回条数一致
    seen = {id(s.result) for s in scored}
    for cand in candidates:
        if id(cand) not in seen:
            scored.append(ScoredCandidate(cand, {k: 0.0 for k in _SCORE_KEYS}, "未获得评分"))
    return scored


ANGLES = (
    "数据反差：用具体数字对比制造冲击",
    "提问开场：用观众自己的困惑起手",
    "痛点直击：先说观众正在受的苦",
    "反常识：先否定一个普遍做法",
    "结果先行：先给最终成果再回溯过程",
    "身份代入：点名某类人群的直接处境",
    "时间成本：强调浪费掉的具体时间",
    "对比框架：把错误做法与正确做法并排",
)


def generate_candidates(
    transcript: str,
    *,
    topic: str = "",
    platform: str = "抖音",
    count: int = 5,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.95,
) -> list[ScoredCandidate]:
    """生成多条候选文案并自评分，按总分降序返回。

    默认用 flash 模型：文案属于短文本创意任务，flash 足够且成本低。
    每条指定不同切入角度，并把已生成的钩子/标题回传，避免同质化。
    """
    if count < 1:
        raise ValueError("count 至少为 1")
    settings = resolve(api_key, base_url, model)

    base_user = USER_TEMPLATE.format(
        topic=topic or "未指定",
        platform=platform,
        transcript=transcript[:12000],
    )

    candidates: list[CopyResult] = []
    for i in range(count):
        angle = ANGLES[i % len(ANGLES)]
        instruction = f"\n\n第 {i + 1} 组方案，切入角度：{angle}。"
        if candidates:
            used = "；".join(f"{c.hook} / {c.title}" for c in candidates)
            instruction += f"\n以下钩子与标题已用过，必须换一种说法，禁止重复：{used}"

        payload = {
            "model": settings.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": base_user + instruction},
            ],
        }
        try:
            data = _post(f"{settings.base_url}/chat/completions", payload, settings.api_key)
            raw = data["choices"][0]["message"]["content"]
            candidates.append(_parse(raw))
        except (LLMError, KeyError, IndexError, ValueError):
            # 单条失败不影响整体；全部失败时下面统一报错
            continue

    if not candidates:
        raise LLMError("所有候选文案生成均失败，请检查网络、密钥与模型名")

    return sorted(
        _score_candidates(candidates, platform=platform, settings=settings),
        key=lambda s: s.total,
        reverse=True,
    )


def best(candidates: list[ScoredCandidate]) -> ScoredCandidate:
    if not candidates:
        raise ValueError("候选列表为空")
    return max(candidates, key=lambda s: s.total)


__all__ = [
    "CopyResult",
    "ConfigError",
    "LLMError",
    "ScoredCandidate",
    "best",
    "generate",
    "generate_candidates",
]
