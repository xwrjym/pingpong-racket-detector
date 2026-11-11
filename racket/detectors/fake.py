"""脚本化的假检测器：不装 ultralytics 也能跑通适配器、评估和测试。"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ..tracker import Detection


class StaticDetector:
    """按帧序号返回预置结果，用于单测和"没有权重"时的链路演练。

    ``script`` 可以是 dict（frame 序号 -> Detection 列表），也可以是
    签名 ``(frame, image_size) -> list[Detection]`` 的函数。
    """

    def __init__(
        self,
        script: dict[int, list[Detection]] | Callable[[Any, tuple[int, int] | None], list[Detection]],
        *,
        names: dict[int, str] | None = None,
    ) -> None:
        self._script = script
        self._index = 0
        self.names = names or {0: "racket_A", 1: "racket_B", 2: "racket_unknown"}
        self.calls = 0

    def detect(self, frame: Any, *, image_size: tuple[int, int] | None = None) -> list[Detection]:
        self.calls += 1
        if callable(self._script):
            return list(self._script(frame, image_size))
        index = self._index
        self._index += 1
        return list(self._script.get(index, []))

    def close(self) -> None:
        return None


class EmptyDetector:
    """永远看不到球拍。"""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: Any, *, image_size: tuple[int, int] | None = None) -> list[Detection]:
        self.calls += 1
        return []

    def close(self) -> None:
        return None
