"""FFmpeg 后端：当前唯一开箱可用的出片路径。"""

from __future__ import annotations

import shutil

from ..ir import Timeline
from ..render import render as ffmpeg_render
from .base import BackendError


class FFmpegBackend:
    name = "ffmpeg"

    def available(self) -> bool:
        return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

    def render(self, timeline: Timeline, output: str, **kwargs) -> str:
        if not self.available():
            raise BackendError("ffmpeg/ffprobe 不在 PATH，请先安装 FFmpeg")
        result = ffmpeg_render(timeline, output, **kwargs)
        return result.output
