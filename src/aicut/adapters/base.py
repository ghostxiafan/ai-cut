"""出片后端适配层。

上层只依赖 Backend 协议；新增后端（剪映明文稿、CapCut、达芬奇）只需实现 render。
"""

from __future__ import annotations

from typing import Protocol

from ..ir import Timeline


class BackendError(RuntimeError):
    """后端不可用或渲染失败。"""


class Backend(Protocol):
    name: str

    def available(self) -> bool: ...

    def render(self, timeline: Timeline, output: str, **kwargs) -> str: ...
