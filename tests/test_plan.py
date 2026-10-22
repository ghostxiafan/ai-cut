from pathlib import Path

import pytest

from aicut.analysis.highlights import Candidate
from aicut.analysis.transcript import Transcript, TranscriptSegment
from aicut.plan import PlanOptions, build_timeline, plan_summary


@pytest.fixture
def transcript() -> Transcript:
    return Transcript(
        segments=[
            TranscriptSegment(0.0, 3.0, "为什么你剪一条视频要四十分钟？"),
            TranscriptSegment(3.0, 6.0, "其实九成时间都浪费在对时间线上。"),
            TranscriptSegment(6.0, 9.0, "把机械劳动交给程序就好。"),
        ]
    )


@pytest.fixture
def candidate() -> Candidate:
    return Candidate(0.0, 9.0, "为什么你剪一条视频要四十分钟？其实九成时间都浪费在对时间线上。把机械劳动交给程序就好。", score=88.0)


def test_builds_video_and_caption_clips(transcript, candidate, tmp_path: Path):
    media = tmp_path / "src.mp4"
    timeline = build_timeline(media, transcript, [candidate], PlanOptions(auto_hook=False, bgm=None))

    assert len(timeline.video) == 1
    assert timeline.video[0].src == str(media.resolve())
    assert timeline.video[0].in_ == 0.0
    assert timeline.video[0].out == 9.0

    captions = [t for t in timeline.text if t.style == "caption"]
    assert len(captions) == 3
    assert captions[1].start == pytest.approx(3.0)
    assert captions[-1].end == pytest.approx(9.0)


def test_hook_dedupes_first_caption(transcript, candidate, tmp_path: Path):
    timeline = build_timeline(tmp_path / "s.mp4", transcript, [candidate], PlanOptions())
    hooks = [t for t in timeline.text if t.style == "hook"]
    assert len(hooks) == 1
    assert hooks[0].start == 0.0
    # 与钩子重复的第一句字幕被去掉，其余保留
    assert len([t for t in timeline.text if t.style == "caption"]) == 2


def test_custom_hook_overrides(transcript, candidate, tmp_path: Path):
    timeline = build_timeline(
        tmp_path / "s.mp4", transcript, [candidate], PlanOptions(hook="别再手剪了")
    )
    hooks = [t for t in timeline.text if t.style == "hook"]
    assert hooks[0].text == "别再手剪了"
    assert len([t for t in timeline.text if t.style == "caption"]) == 3


def test_no_hook_option(transcript, candidate, tmp_path: Path):
    timeline = build_timeline(
        tmp_path / "s.mp4", transcript, [candidate], PlanOptions(auto_hook=False)
    )
    assert not [t for t in timeline.text if t.style == "hook"]


def test_bgm_loops_to_full_length(transcript, candidate, tmp_path: Path):
    bgm = tmp_path / "bgm.m4a"
    timeline = build_timeline(
        tmp_path / "s.mp4", transcript, [candidate], PlanOptions(bgm=str(bgm), bgm_volume=0.2)
    )
    assert len(timeline.audio) == 1
    assert timeline.audio[0].loop is True
    assert timeline.audio[0].out == pytest.approx(9.0)
    assert timeline.audio[0].volume == 0.2


def test_multiple_candidates_offset_text(transcript, tmp_path: Path):
    first = Candidate(0.0, 6.0, "a", score=90)
    second = Candidate(6.0, 9.0, "b", score=80)
    timeline = build_timeline(tmp_path / "s.mp4", transcript, [first, second], PlanOptions(auto_hook=False))
    assert len(timeline.video) == 2
    assert timeline.video[1].in_ == pytest.approx(6.0)
    later = [t for t in timeline.text if t.start >= 6.0]
    assert later
    assert later[0].start == pytest.approx(6.0)


def test_long_caption_split(transcript, candidate, tmp_path: Path):
    timeline = build_timeline(
        tmp_path / "s.mp4", transcript, [candidate], PlanOptions(auto_hook=False, max_chars_per_cue=5)
    )
    captions = [t for t in timeline.text if t.style == "caption"]
    assert len(captions) > 3
    for cue in captions:
        assert cue.end > cue.start


def test_no_candidates_raises(transcript, tmp_path: Path):
    with pytest.raises(ValueError):
        build_timeline(tmp_path / "s.mp4", transcript, [])


def test_plan_summary_shape(transcript, candidate, tmp_path: Path):
    timeline = build_timeline(tmp_path / "s.mp4", transcript, [candidate], PlanOptions())
    summary = plan_summary([candidate], timeline)
    assert summary["clips"][0]["score"] == 88.0
    assert summary["timeline"]["video_clips"] == 1
