"""球拍观测与击球候选的统一数据契约（racket.v1）。

字段与《球拍检测模型独立方案》第 6、7 节一致。本模块刻意不 import
``tt_referee``：球拍模型要能独立训练、独立部署、独立回滚，只靠这一份契约
与适配器对接。

两条校验原则（与现有 BallObservation 一致）：
* ``visible=False`` 时不允许带坐标，避免把预测位置当成真实检测；
* 坐标必须是原始图像像素、且在图像范围内，时间戳必须属于被预测的那一帧。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

SCHEMA_VERSION = "racket.v1"

SIDES: tuple[str, ...] = ("A", "B", "UNKNOWN")
#: predicted = 该帧没有检测，由跟踪器外推补位（低置信证据，下游应降权）
STATUSES: tuple[str, ...] = ("ok", "occluded", "lost", "low_conf", "predicted")
DECISIONS: tuple[str, ...] = ("candidate", "confirmed", "rejected", "pending")

Side = Literal["A", "B", "UNKNOWN"]
Status = Literal["ok", "occluded", "lost", "low_conf", "predicted"]
BBox = tuple[float, float, float, float]

#: 判定一个 bbox 是否"在图像内"时允许的亚像素误差
_EDGE_TOL = 0.5


class SchemaError(ValueError):
    """观测或候选不符合 racket.v1 契约。"""


def check_bbox(bbox: Any, image_size: tuple[int, int]) -> BBox:
    """校验并归一化 bbox，返回 (x1, y1, x2, y2) 浮点元组。"""
    try:
        values = [float(v) for v in bbox]
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"bbox_xyxy 必须是 4 个数字，收到 {bbox!r}") from exc
    if len(values) != 4:
        raise SchemaError(f"bbox_xyxy 必须是 4 个数字，收到 {bbox!r}")
    x1, y1, x2, y2 = values
    if not (x2 > x1 and y2 > y1):
        raise SchemaError(f"bbox_xyxy 需要 x2>x1 且 y2>y1，收到 {bbox!r}")
    width, height = image_size
    if min(values) < -_EDGE_TOL or x2 > width + _EDGE_TOL or y2 > height + _EDGE_TOL:
        raise SchemaError(
            f"bbox_xyxy {bbox!r} 超出图像范围 {image_size}，"
            "坐标必须还原到原始图像"
        )
    return (x1, y1, x2, y2)


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class RacketObservation:
    """单帧单支球拍的观测。

    ``visible=False`` 表示这一帧没看到球拍（被遮挡或出画），此时所有坐标与
    置信度都必须为 None——下游据此走"漏检/遮挡"分支，而不是拿到一个假位置。
    """

    session_id: str
    frame_id: int
    timestamp_s: float
    side: Side = "UNKNOWN"
    bbox_xyxy: BBox | None = None
    confidence: float | None = None
    track_id: str | None = None
    status: Status = "ok"
    visible: bool = True
    source: str = "racket_model_a"
    model_version: str = "racket_v0.1"
    schema_version: str = SCHEMA_VERSION
    image_size: tuple[int, int] = (1920, 1080)

    # ---------------------------------------------------------------- 校验
    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(
                f"schema_version 必须是 {SCHEMA_VERSION!r}，收到 {self.schema_version!r}"
            )
        if not isinstance(self.session_id, str) or not self.session_id:
            raise SchemaError("session_id 不能为空")
        if self.frame_id < 0:
            raise SchemaError(f"frame_id 不能为负，收到 {self.frame_id}")
        if self.timestamp_s < 0:
            raise SchemaError(f"timestamp_s 不能为负，收到 {self.timestamp_s}")
        if self.side not in SIDES:
            raise SchemaError(f"side 必须是 {SIDES} 之一，收到 {self.side!r}")
        if self.status not in STATUSES:
            raise SchemaError(f"status 必须是 {STATUSES} 之一，收到 {self.status!r}")
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise SchemaError(f"image_size 非法：{self.image_size!r}")

        if self.visible:
            if self.bbox_xyxy is None:
                raise SchemaError("visible=True 时必须给出 bbox_xyxy")
            object.__setattr__(
                self, "bbox_xyxy", check_bbox(self.bbox_xyxy, self.image_size)
            )
            if self.confidence is None:
                raise SchemaError("visible=True 时必须给出 confidence")
            if not 0.0 <= float(self.confidence) <= 1.0:
                raise SchemaError(f"confidence 必须在 0~1，收到 {self.confidence}")
        else:
            if self.bbox_xyxy is not None or self.confidence is not None:
                raise SchemaError(
                    "visible=False 时不能带 bbox 或 confidence（避免把预测位置当成真实检测）"
                )
            if self.status == "ok":
                raise SchemaError("visible=False 时 status 不能是 'ok'")

    # ---------------------------------------------------------------- 便捷
    @property
    def center_px(self) -> tuple[float, float] | None:
        return bbox_center(self.bbox_xyxy) if self.bbox_xyxy else None

    @property
    def bbox_area(self) -> float | None:
        if not self.bbox_xyxy:
            return None
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x2 - x1) * (y2 - y1)

    def to_dict(self) -> dict[str, Any]:
        """按方案第 6 节的字段顺序输出，便于和文档逐字段对照。"""
        center = self.center_px
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "timestamp_s": round(float(self.timestamp_s), 6),
            "source": self.source,
            "model_version": self.model_version,
            "visible": self.visible,
            "side": self.side,
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy else None,
            "center_px": list(center) if center else None,
            "confidence": None if self.confidence is None else round(float(self.confidence), 4),
            "track_id": self.track_id,
            "status": self.status,
            "image_size": list(self.image_size),
        }

    @classmethod
    def lost(
        cls,
        session_id: str,
        frame_id: int,
        timestamp_s: float,
        side: Side = "UNKNOWN",
        track_id: str | None = None,
        status: Status = "lost",
        **kwargs: Any,
    ) -> "RacketObservation":
        return cls(
            session_id=session_id,
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            side=side,
            visible=False,
            status=status,
            track_id=track_id,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RacketObservation":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        payload.pop("center_px", None)  # 派生字段
        bbox = payload.get("bbox_xyxy")
        if bbox is not None:
            payload["bbox_xyxy"] = tuple(float(v) for v in bbox)
        size = payload.get("image_size")
        if size is not None:
            payload["image_size"] = (int(size[0]), int(size[1]))
        return cls(**payload)


@dataclass(frozen=True)
class HitCandidate:
    """击球候选：球拍证据 + 球轨迹证据的融合结果，不是最终判分。"""

    event_id: str
    timestamp_s: float
    side: Side = "UNKNOWN"
    type: str = "HIT_CANDIDATE"
    sources: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    quality_score: float = 0.0
    decision: str = "candidate"

    def __post_init__(self) -> None:
        if not self.event_id:
            raise SchemaError("event_id 不能为空")
        if self.timestamp_s < 0:
            raise SchemaError("timestamp_s 不能为负")
        if self.side not in SIDES:
            raise SchemaError(f"side 必须是 {SIDES} 之一，收到 {self.side!r}")
        if self.decision not in DECISIONS:
            raise SchemaError(f"decision 必须是 {DECISIONS} 之一，收到 {self.decision!r}")
        if not 0.0 <= float(self.quality_score) <= 1.0:
            raise SchemaError(f"quality_score 必须在 0~1，收到 {self.quality_score}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp_s": round(float(self.timestamp_s), 6),
            "type": self.type,
            "side": self.side,
            "sources": list(self.sources),
            "evidence": list(self.evidence),
            "quality_score": round(float(self.quality_score), 4),
            "decision": self.decision,
        }


class ObservationLog:
    """JSONL 观测日志：每条一行，断电/崩溃后可逐行回放。"""

    def __init__(self, path: str | Path, *, truncate: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w" if truncate else "a", encoding="utf-8")

    def write(self, obs: RacketObservation) -> None:
        self._fh.write(json.dumps(obs.to_dict(), ensure_ascii=False) + "\n")

    def write_many(self, observations: Iterable[RacketObservation]) -> None:
        for obs in observations:
            self.write(obs)

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "ObservationLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_observations(path: str | Path) -> Iterator[RacketObservation]:
    """逐行读回观测日志；坏行直接跳过（崩溃时最后一行可能写了一半）。"""
    with open(Path(path), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield RacketObservation.from_dict(json.loads(line))
            except (json.JSONDecodeError, SchemaError, TypeError, KeyError):
                continue
