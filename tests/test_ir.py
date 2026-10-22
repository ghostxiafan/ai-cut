import json
from pathlib import Path

import pytest

from aicut.ir import IRValidationError, Timeline, load_timeline, timeline_from_dict


def test_relative_paths_resolve_against_ir_dir(tmp_path: Path):
    ir_file = tmp_path / "timeline.json"
    ir_file.write_text(
        json.dumps(
            {
                "canvas": {"w": 720, "h": 1280, "fps": 25},
                "tracks": {
                    "video": [{"src": "clips/a.mp4", "in": 1.0, "out": 4.0}],
                    "audio": [{"src": "bgm.m4a", "out": 3.0, "volume": 0.2, "loop": True}],
                    "text": [{"text": "你好", "start": 0, "end": 2, "style": "hook"}],
                },
            }
        ),
        encoding="utf-8",
    )

    timeline = load_timeline(ir_file)
    assert timeline.canvas.w == 720
    assert timeline.video[0].src == str((tmp_path / "clips" / "a.mp4").resolve())
    assert timeline.video[0].duration == 3.0
    assert timeline.audio[0].loop is True
    assert timeline.text[0].style == "hook"
    assert timeline.duration == 3.0


def test_duration_falls_back_to_audio_when_no_video(tmp_path: Path):
    timeline = timeline_from_dict({"tracks": {"audio": [{"src": str(tmp_path / "a.mp3"), "out": 7.0}]}})
    assert timeline.video == []
    assert timeline.duration == 7.0


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"tracks": {"video": [{"src": "a.mp4", "in": 5, "out": 2}]}},
        {"tracks": {"video": [{"out": 2}]}},
        {"tracks": {"text": [{"text": "x", "start": 3, "end": 1}]}, "video": [{"src": "a.mp4", "out": 1}]},
        {"canvas": {"w": 0, "h": 100}},
    ],
)
def test_invalid_ir_rejected(payload: dict):
    with pytest.raises(IRValidationError):
        timeline_from_dict(payload)


def test_roundtrip(tmp_path: Path):
    timeline = timeline_from_dict(
        {
            "canvas": {"w": 1080, "h": 1920, "fps": 30},
            "tracks": {"video": [{"src": str(tmp_path / "a.mp4"), "out": 2.5, "muted": True}]},
        }
    )
    data = json.loads(timeline.to_json())
    again = timeline_from_dict(data)
    assert again.to_dict() == data
    assert isinstance(again, Timeline)
