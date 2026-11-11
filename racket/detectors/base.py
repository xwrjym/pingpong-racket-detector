"""检测后端公共部分：配置与侧别解析。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from ..tracker import Detection


@dataclass
class DetectorConfig:
    conf_thr: float = 0.25
    iou_thr: float = 0.50
    imgsz: int = 640
    #: "auto" 交给后端自己挑；也可写 "cpu" / "cuda:0"
    device: str = "auto"
    half: bool = False
    max_det: int = 4


class Detector(Protocol):
    def detect(self, frame: Any, *, image_size: tuple[int, int] | None = None) -> list[Detection]:
        ...

    def close(self) -> None:
        ...


def side_from_class_name(name: str) -> str:
    """``racket_A`` -> ``A``；``racket_unknown`` -> ``UNKNOWN``。"""
    if not name:
        return "UNKNOWN"
    tail = name.rsplit("_", 1)[-1].upper()
    if tail in ("A", "B"):
        return tail
    return "UNKNOWN"


def boxes_to_detections(
    xyxy: Sequence[Sequence[float]],
    confidences: Sequence[float],
    class_ids: Sequence[int],
    names: dict[int, str] | None = None,
) -> list[Detection]:
    out: list[Detection] = []
    for box, conf, cls in zip(xyxy, confidences, class_ids):
        name = (names or {}).get(int(cls), "")
        out.append(Detection(
            bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            confidence=float(conf),
            side=side_from_class_name(name),
            class_id=int(cls),
            class_name=name or None,
        ))
    return out
