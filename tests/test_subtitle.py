from pathlib import Path

from aicut.subtitle import Cue, write_ass, write_srt


def test_srt_format_and_encoding(tmp_path: Path):
    out = write_srt([Cue(0, 1.5, "第一行"), Cue(1.5, 3.0, "第二行\n换行")], tmp_path / "a.srt")
    text = out.read_text(encoding="utf-8-sig")
    assert "1\n00:00:00,000 --> 00:00:01,500\n第一行" in text
    assert "2\n00:00:01,500 --> 00:00:03,000" in text
    assert text.rstrip().endswith("换行")


def test_ass_maps_styles_and_escapes_newlines(tmp_path: Path):
    out = write_ass(
        [Cue(0, 1, "钩子", "hook"), Cue(1, 2, "字幕\n第二行", "caption")],
        tmp_path / "a.ass",
        width=1080,
        height=1920,
    )
    text = out.read_text(encoding="utf-8-sig")
    assert "PlayResX: 1080" in text
    assert "Style: Hook," in text
    assert "Style: Caption," in text
    assert ",Hook,,0,0,0,,钩子" in text
    assert r"字幕\N第二行" in text


def test_ass_custom_style_mapping(tmp_path: Path):
    out = write_ass([Cue(0, 1, "x", "title")], tmp_path / "b.ass", 720, 1280, styles={"title": "Hook"})
    assert ",Hook,,0,0,0,,x" in out.read_text(encoding="utf-8-sig")
