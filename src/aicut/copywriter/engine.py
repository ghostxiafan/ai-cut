"""文案引擎：根据转写文本生成钩子、标题、口播稿与标签。

只依赖标准库（urllib），兼容任意 OpenAI 兼容的 /chat/completions 接口，
因此 DeepSeek / Qwen / 本地 vLLM 都能直接用。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

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
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
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


def _parse(raw: str) -> CopyResult:
    data = json.loads(_strip_code_fence(raw))
    return CopyResult(
        hook=str(data.get("hook", "")),
        title=str(data.get("title", "")),
        oral_script=str(data.get("oral_script", "")),
        chapters=[str(x) for x in data.get("chapters", []) or []],
        hashtags=[str(x) for x in data.get("hashtags", []) or []],
        cover_text=str(data.get("cover_text", "")),
        comment_bait=str(data.get("comment_bait", "")),
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
    api_key = api_key or os.environ.get("AICUT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = (base_url or os.environ.get("AICUT_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    model = model or os.environ.get("AICUT_LLM_MODEL") or "gpt-4o-mini"
    if not api_key:
        raise LLMError("缺少 API Key：设置环境变量 AICUT_LLM_API_KEY 或传入 api_key")

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(topic=topic or "未指定", platform=platform, transcript=transcript[:12000]),
            },
        ],
    }
    data = _post(f"{base_url}/chat/completions", payload, api_key)
    try:
        raw = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"响应结构异常: {json.dumps(data)[:400]}") from exc
    return _parse(raw)
