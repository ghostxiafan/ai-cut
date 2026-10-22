"""各平台发布文案规格。

平台之间不是"换个标题"那么简单：字数上限、语气、标签习惯都不同，
必须分开生成，否则抖音的口语标题搬到 B 站会显得随意，小红书的 emoji
正文搬到视频号又显得不像官方号。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformSpec:
    name: str
    title_max: int
    body_max: int
    tags_max: int
    style: str
    body_hint: str


PLATFORMS: dict[str, PlatformSpec] = {
    "抖音": PlatformSpec(
        name="抖音",
        title_max=30,
        body_max=120,
        tags_max=5,
        style="口语、直接、带情绪。标题前 12 字必须是钩子，不要书面语。",
        body_hint="正文一两句话即可，补足标题没说完的信息，结尾可抛一个问题引导评论。",
    ),
    "小红书": PlatformSpec(
        name="小红书",
        title_max=20,
        body_max=800,
        tags_max=10,
        style="像跟姐妹分享，标题带情绪或结果，可用 emoji，不要硬广感。",
        body_hint="正文分段，每段短句，可用 1-3 个 emoji 作段落标记，强调可复制的做法。",
    ),
    "视频号": PlatformSpec(
        name="视频号",
        title_max=16,
        body_max=1000,
        tags_max=3,
        style="偏正式、朴实、可信，面向熟人社交，少用网络热词。",
        body_hint="正文平实陈述，讲清楚价值和适用人群，避免过度夸张。",
    ),
    "B站": PlatformSpec(
        name="B站",
        title_max=80,
        body_max=1000,
        tags_max=10,
        style="可以有梗，标题信息量要大，允许长标题说明剧情或结论。",
        body_hint="正文写简介：背景、要点、结论，可用分点，适合配合章节时间轴。",
    ),
    "快手": PlatformSpec(
        name="快手",
        title_max=30,
        body_max=200,
        tags_max=5,
        style="接地气、强调实用和真实经验，少用抽象词。",
        body_hint="正文讲清具体做法和适用场景，语气像邻里分享。",
    ),
}

DEFAULT_PLATFORMS = ("抖音", "小红书", "视频号", "B站")


def spec_for(name: str) -> PlatformSpec:
    """未知平台按抖音规格兜底，避免直接报错中断出片。"""
    return PLATFORMS.get(name, PLATFORMS["抖音"])


def normalize_platforms(names: list[str] | tuple[str, ...] | None) -> list[str]:
    if not names:
        return list(DEFAULT_PLATFORMS)
    out: list[str] = []
    for raw in names:
        name = str(raw).strip()
        if not name:
            continue
        if name not in out:
            out.append(name)
    return out or list(DEFAULT_PLATFORMS)
