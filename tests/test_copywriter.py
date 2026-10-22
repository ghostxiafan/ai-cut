"""文案引擎测试：全部走假的 HTTP 层，不发真实请求。"""

import json

import pytest

from aicut.copywriter import engine
from aicut.copywriter.engine import (
    CopyResult,
    LLMError,
    ScoredCandidate,
    _extract_json_list,
    _parse,
    _strip_code_fence,
    best,
    generate,
    generate_candidates,
)

GOOD = {
    "hook": "剪一条视频40分钟？",
    "cover_text": "40→6分钟",
    "oral_script": "开头。中间。结尾。",
    "chapters": ["痛点", "对比"],
    "platforms": {
        "抖音": {"title": "6分钟剪完一条片", "body": "别再手剪了。", "hashtags": ["AI剪辑", "#剪辑"]},
        "小红书": {"title": "剪片效率翻倍", "body": "姐妹们，这个流程真的省时间", "hashtags": ["#剪辑"]},
        "视频号": {"title": "剪辑效率提升", "body": "分享一个实用流程", "hashtags": ["#剪辑"]},
        "B站": {"title": "我把剪辑流程自动化了，40分钟变6分钟", "body": "简介正文", "hashtags": ["#剪辑"]},
    },
}


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _score_json(rows: list[dict]) -> str:
    return json.dumps(rows, ensure_ascii=False)


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch):
    """不读取真实 .env，模型名固定为默认值。"""
    from aicut import config

    for var in ("AICUT_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
                "AICUT_LLM_BASE_URL", "AICUT_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config, "_dotenv_candidates", lambda: [])
    monkeypatch.setattr(config, "_env_loaded", False, raising=False)
    yield


@pytest.fixture
def fake_post(monkeypatch):
    """记录请求，按顺序返回预置响应。"""
    calls: list[dict] = []
    queue: list[dict] = []

    def _post(url, payload, api_key, timeout=180):
        calls.append({"url": url, "payload": payload, "api_key": api_key})
        if not queue:
            raise AssertionError("没有更多预置响应")
        return queue.pop(0)

    monkeypatch.setattr(engine, "_post", _post)
    monkeypatch.setenv("AICUT_LLM_API_KEY", "sk-test")
    return {"calls": calls, "queue": queue}


def test_strip_code_fence_variants():
    assert _strip_code_fence('```json\n{"a":1}\n```') == '{"a":1}'
    assert _strip_code_fence('```\n{"a":1}\n```') == '{"a":1}'
    assert _strip_code_fence('{"a":1}') == '{"a":1}'


def test_parse_maps_all_fields():
    result = _parse(json.dumps(GOOD, ensure_ascii=False), ["抖音", "小红书"])
    assert result.hook == GOOD["hook"]
    assert result.cover_text == GOOD["cover_text"]
    douyin = result.platform("抖音")
    assert douyin is not None
    assert douyin.title == "6分钟剪完一条片"
    assert douyin.body == "别再手剪了。"
    assert douyin.hashtags == ["#AI剪辑", "#剪辑"]
    assert result.to_dict()["publish"][1]["platform"] == "小红书"


def test_parse_handles_missing_keys():
    result = _parse('{"hook":"只有钩子"}', ["抖音"])
    assert result.hook == "只有钩子"
    assert result.cover_text == ""
    douyin = result.platform("抖音")
    assert douyin is not None
    assert douyin.title == ""
    assert douyin.note == "该平台未产出，请重试或减少平台数量"


def test_parse_normalizes_hashtags():
    result = _parse(
        json.dumps(
            {
                "hook": "h",
                "platforms": {
                    "抖音": {
                        "title": "t",
                        "body": "b",
                        "hashtags": ["AI剪辑", "#AI剪辑", " 短视频 剪辑 ", "", "剪辑效率"],
                    }
                },
            },
            ensure_ascii=False,
        ),
        ["抖音"],
    )
    assert result.platform("抖音").hashtags == ["#AI剪辑", "#短视频剪辑", "#剪辑效率"]


def test_platform_limits_are_enforced():
    result = _parse(
        json.dumps(
            {
                "hook": "h",
                "platforms": {
                    "视频号": {
                        "title": "一" * 40,  # 上限 16
                        "body": "正" * 1200,  # 上限 1000
                        "hashtags": [f"#t{i}" for i in range(10)],  # 上限 3
                    }
                },
            },
            ensure_ascii=False,
        ),
        ["视频号"],
    )
    vh = result.platform("视频号")
    assert len(vh.title) == 16
    assert len(vh.body) == 1000
    assert len(vh.hashtags) == 3
    assert "标题已截断至 16 字" in vh.note
    assert "正文已截断至 1000 字" in vh.note


def test_platform_array_format_supported():
    payload = json.dumps(
        {
            "hook": "h",
            "platforms": [{"platform": "B站", "title": "标题", "body": "简介", "hashtags": ["#a"]}],
        },
        ensure_ascii=False,
    )
    result = _parse(payload, ["B站"])
    assert result.platform("B站").title == "标题"


def test_unknown_platform_falls_back_to_douyin_spec():
    from aicut.copywriter.platforms import spec_for

    assert spec_for("知乎").title_max == spec_for("抖音").title_max


def test_extract_json_list_ignores_surrounding_text():
    rows = _extract_json_list('说明文字\n[{"index":0}]\n尾部')
    assert rows == [{"index": 0}]


def test_extract_json_list_rejects_non_json():
    with pytest.raises(LLMError):
        _extract_json_list("没有数组")


def test_generate_single_uses_flash_by_default(fake_post):
    fake_post["queue"].append(_chat_response(json.dumps(GOOD, ensure_ascii=False)))
    result = generate("测试文本")
    assert result.hook == GOOD["hook"]
    assert fake_post["calls"][0]["payload"]["model"] == "deepseek-flash"


def test_generate_candidates_scores_and_sorts(fake_post):
    variants = [
        {**GOOD, "hook": "A"},
        {**GOOD, "hook": "B"},
        {**GOOD, "hook": "C"},
    ]
    for v in variants:
        fake_post["queue"].append(_chat_response(json.dumps(v, ensure_ascii=False)))

    scores = [
        {"index": 0, "hook": 10, "emotion": 10, "clarity": 10, "platform": 10, "finish": 10, "note": "中"},
        {"index": 1, "hook": 19, "emotion": 18, "clarity": 17, "platform": 16, "finish": 15, "note": "好"},
        {"index": 2, "hook": 12, "emotion": 11, "clarity": 13, "platform": 12, "finish": 11, "note": "偏弱"},
    ]
    fake_post["queue"].append(_chat_response(_score_json(scores)))

    scored = generate_candidates("文本", count=3)

    assert [s.result.hook for s in scored] == ["B", "C", "A"]
    assert scored[0].total == 85
    assert scored[0].note == "好"
    assert len(fake_post["calls"]) == 4  # 3 次生成 + 1 次评分
    assert all(c["payload"]["model"] == "deepseek-flash" for c in fake_post["calls"])


def test_candidate_prompt_varies_per_item(fake_post):
    fake_post["queue"].append(_chat_response(json.dumps(GOOD, ensure_ascii=False)))
    fake_post["queue"].append(_chat_response(json.dumps(GOOD, ensure_ascii=False)))
    fake_post["queue"].append(
        _chat_response(
            _score_json([
                {"index": 0, "hook": 1, "emotion": 1, "clarity": 1, "platform": 1, "finish": 1},
                {"index": 1, "hook": 2, "emotion": 2, "clarity": 2, "platform": 2, "finish": 2},
            ])
        )
    )

    generate_candidates("文本", count=2)
    first = fake_post["calls"][0]["payload"]["messages"][1]["content"]
    second = fake_post["calls"][1]["payload"]["messages"][1]["content"]
    assert first != second
    assert "第 1 组" in first and "第 2 组" in second


def test_second_candidate_is_told_to_avoid_first(fake_post):
    unique = {**GOOD, "hook": "独特的钩子XYZ", "title": "独特标题XYZ"}
    fake_post["queue"].append(_chat_response(json.dumps(unique, ensure_ascii=False)))
    fake_post["queue"].append(_chat_response(json.dumps({**GOOD, "hook": "另一条"}, ensure_ascii=False)))
    fake_post["queue"].append(
        _chat_response(_score_json([{"index": i, "hook": 5} for i in range(2)]))
    )

    generate_candidates("文本", count=2)
    second = fake_post["calls"][1]["payload"]["messages"][1]["content"]
    assert "独特的钩子XYZ" in second
    assert "禁止重复" in second


def test_survives_partial_generation_failure(fake_post):
    fake_post["queue"].append({"error": "boom"})  # 结构异常，被吞掉
    fake_post["queue"].append(_chat_response(json.dumps(GOOD, ensure_ascii=False)))
    fake_post["queue"].append(
        _chat_response(
            _score_json([{"index": 0, "hook": 5, "emotion": 5, "clarity": 5, "platform": 5, "finish": 5}])
        )
    )

    scored = generate_candidates("文本", count=2)
    assert len(scored) == 1
    assert scored[0].total == 25


def test_all_generation_failures_raise(fake_post):
    fake_post["queue"].append({"error": "boom"})
    with pytest.raises(LLMError):
        generate_candidates("文本", count=1)


def test_missing_scores_are_filled_with_zero(fake_post):
    fake_post["queue"].append(_chat_response(json.dumps(GOOD, ensure_ascii=False)))
    fake_post["queue"].append(_chat_response("[{\"index\": 0, \"hook\": 20}]"))

    scored = generate_candidates("文本", count=1)
    assert scored[0].scores["hook"] == 20
    assert scored[0].scores["finish"] == 0
    assert scored[0].total == 20


def test_scores_are_clamped(fake_post):
    fake_post["queue"].append(_chat_response(json.dumps(GOOD, ensure_ascii=False)))
    fake_post["queue"].append(
        _chat_response('[{"index":0,"hook":99,"emotion":-5,"clarity":10,"platform":10,"finish":10}]')
    )
    scored = generate_candidates("文本", count=1)
    assert scored[0].scores["hook"] == 20
    assert scored[0].scores["emotion"] == 0


def test_duplicate_index_rows_ignored(fake_post):
    fake_post["queue"].append(_chat_response(json.dumps(GOOD, ensure_ascii=False)))
    fake_post["queue"].append(
        _chat_response('[{"index":0,"hook":20,"emotion":20,"clarity":20,"platform":20,"finish":20},'
                       '{"index":0,"hook":1,"emotion":1,"clarity":1,"platform":1,"finish":1},'
                       '{"index":99,"hook":5}]')
    )
    scored = generate_candidates("文本", count=1)
    assert len(scored) == 1
    assert scored[0].total == 100


def test_count_must_be_positive():
    with pytest.raises(ValueError):
        generate_candidates("文本", count=0)


def test_best_selects_highest():
    low = ScoredCandidate(CopyResult(hook="low"), {"hook": 1})
    high = ScoredCandidate(CopyResult(hook="high"), {"hook": 9})
    assert best([low, high]).result.hook == "high"
    with pytest.raises(ValueError):
        best([])
