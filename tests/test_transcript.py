from pathlib import Path

import pytest

from aicut.analysis.transcript import (
    Transcript,
    TranscriptSegment,
    load_transcript,
    parse_srt,
    transcript_from_text,
)


def test_plain_text_and_timestamps():
    t = Transcript(segments=[TranscriptSegment(0, 1.2, "你好"), TranscriptSegment(1.2, 3.0, "世界")])
    assert t.plain_text() == "你好世界"
    assert t.duration == 3.0
    assert t.plain_text(with_timestamps=True).splitlines()[1] == "[00:01] 世界"


def test_srt_roundtrip(tmp_path: Path):
    t = Transcript(segments=[TranscriptSegment(0.5, 2.25, "第一句"), TranscriptSegment(2.25, 4.0, "第二句")])
    srt = t.to_srt(tmp_path / "a.srt")
    again = parse_srt(srt.read_text(encoding="utf-8-sig"))
    assert len(again) == 2
    assert again.segments[0].start == pytest.approx(0.5)
    assert again.segments[1].end == pytest.approx(4.0)
    assert again.segments[1].text == "第二句"


def test_parse_srt_handles_crlf_and_dot_ms():
    text = "1\r\n00:00:01.500 --> 00:00:03.000\r\nA\r\n\r\n2\r\n00:00:03,000 --> 00:00:05,250\r\nB\r\n"
    t = parse_srt(text)
    assert [s.text for s in t.segments] == ["A", "B"]
    assert t.segments[0].start == pytest.approx(1.5)
    assert t.segments[1].end == pytest.approx(5.25)


def test_slice_clips_to_window():
    t = Transcript(segments=[TranscriptSegment(0, 5, "aa"), TranscriptSegment(5, 10, "bb")])
    seg = t.slice(2.0, 7.0)
    assert len(seg) == 2
    assert seg[0].start == pytest.approx(0.0)
    assert seg[0].end == pytest.approx(3.0)
    assert seg[1].start == pytest.approx(3.0)
    assert seg[1].end == pytest.approx(5.0)


def test_text_estimates_timings():
    t = transcript_from_text("第一句话。第二句话！第三句话？")
    assert len(t) == 3
    assert t.segments[0].start == 0.0
    assert t.segments[-1].end == pytest.approx(t.duration)
    assert t.duration > 2.0


def test_load_by_extension(tmp_path: Path):
    json_path = tmp_path / "t.json"
    Transcript(segments=[TranscriptSegment(0, 1, "x")]).save(json_path)
    assert len(load_transcript(json_path)) == 1

    txt_path = tmp_path / "t.txt"
    txt_path.write_text("你好。世界！", encoding="utf-8")
    assert len(load_transcript(txt_path)) == 2

    with pytest.raises(ValueError):
        load_transcript(tmp_path / "t.unsupported")
