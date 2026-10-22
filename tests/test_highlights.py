from aicut.analysis import highlights
from aicut.analysis.highlights import (
    Candidate,
    build_candidates,
    rank_candidates,
    score_candidate,
    select,
)
from aicut.analysis.transcript import Transcript, TranscriptSegment


def _transcript(lines: list[tuple[float, float, str]]) -> Transcript:
    return Transcript(segments=[TranscriptSegment(s, e, t) for s, e, t in lines])


def test_hook_rich_opening_scores_higher():
    weak = score_candidate(
        Candidate(0, 20, "今天我们来讲一些关于视频剪辑的事情，内容比较多，大家慢慢看下去就知道了。"),
        Transcript(),
        20,
    )
    strong = score_candidate(
        Candidate(0, 20, "为什么你剪一条视频要四十分钟？其实九成时间都浪费在对时间线上。"),
        Transcript(),
        20,
    )
    assert strong.score > weak.score
    assert strong.reasons["钩子"] > weak.reasons["钩子"]


def test_candidate_durations_respect_bounds():
    lines = [(i * 5.0, (i + 1) * 5.0, f"第{i}句内容。") for i in range(12)]
    candidates = build_candidates(_transcript(lines), target=20, min_duration=10, max_duration=30)
    assert candidates
    assert all(10 <= c.duration <= 30 for c in candidates)


def test_rank_avoids_overlap():
    a = Candidate(0, 30, "aaa", score=90)
    b = Candidate(5, 35, "bbb", score=95)
    c = Candidate(80, 110, "ccc", score=80)
    picked = rank_candidates([a, b, c], top=2, min_gap=3)
    assert len(picked) == 2
    assert all(not (x.start < y.end and y.start < x.end) for x, y in zip(picked, picked[1:]))


def test_select_returns_best_and_sorted_by_time():
    lines = [
        (0, 10, "大家好，今天随便聊聊。"),
        (10, 20, "其实真正拖慢你的是找素材。"),
        (20, 30, "为什么有人一小时能出三条片？"),
        (30, 60, "因为他把机械劳动交给程序，只保留判断。"),
    ]
    picked = select(_transcript(lines), target=20, max_duration=40, top=2)
    assert 1 <= len(picked) <= 2
    assert all(picked[i].start <= picked[i + 1].start for i in range(len(picked) - 1))


def test_uniform_candidates_covers_timeline():
    cands = highlights.uniform_candidates(75.0, target=30.0, top=1)
    assert len(cands) == 1
    assert cands[0].start == 0.0
    assert cands[0].end == 30.0
    assert cands[0].text == ""


def test_uniform_candidates_respects_duration():
    cands = highlights.uniform_candidates(20.0, target=30.0, top=1)
    assert len(cands) == 1
    assert cands[0].end == 20.0


def test_uniform_candidates_multiple_segments_do_not_overlap():
    cands = highlights.uniform_candidates(300.0, target=20.0, top=3)
    assert len(cands) == 3
    for a, b in zip(cands, cands[1:]):
        assert a.end <= b.start
    assert all(19 <= c.duration <= 21 for c in cands)


def test_uniform_candidates_handles_too_short():
    assert highlights.uniform_candidates(2.0, target=30.0) == []
    assert highlights.uniform_candidates(0.0) == []


def test_empty_transcript_yields_no_candidates():
    assert select(Transcript(), target=30) == []
