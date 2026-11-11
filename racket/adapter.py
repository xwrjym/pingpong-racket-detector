"""球拍模型统一适配器。

职责边界（方案 §8）：把任意检测后端的结果翻译成 ``racket.v1`` 观测，
并在出口处做坐标、时间与侧别校验。**适配器不判分**，也不 import 计分状态机。

同时实现方案 §2 的「只在 Proposal 区间内推理」：``ProposalGate`` 决定当前
时间戳要不要唤醒模型，非区间直接返回空列表，省掉低成本设备上的空转开销。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Protocol, Sequence

from .geometry import bbox_center, clip_bbox, iou, side_from_x
from .schema import ObservationLog, RacketObservation, SchemaError
from .tracker import Detection, RacketTracker, TrackerConfig


class Detector(Protocol):
    """任何能给出候选框的后端。实现见 ``detectors.yolo`` / ``detectors.onnx``。"""

    def detect(self, frame: Any, *, image_size: tuple[int, int] | None = None) -> list[Detection]:
        ...

    def close(self) -> None:
        ...


@dataclass
class AdapterConfig:
    session_id: str
    source: str = "racket_model_a"
    model_version: str = "racket_v0.1"
    #: 低于该置信度的框直接丢弃（不进跟踪器，避免闪断）
    conf_thr: float = 0.25
    #: 球台网线在图像中的横向位置比例；0.5 表示画面正中
    net_x_ratio: float = 0.5
    #: 距网线 ±该比例的画面宽度内判 UNKNOWN，不猜侧别
    dead_zone_ratio: float = 0.02
    #: 类别 id -> 侧别。默认为空：单类别模型的 class 0 是"球拍"，不是"A 侧"，
    #: 只有真的训练了 racket_A/racket_B 三类别模型时才需要填这个映射。
    class_side_map: dict[int, str] = field(default_factory=dict)
    #: 每帧最多保留几支球拍（业余场景通常 1~2 支）
    max_boxes: int = 2
    #: 每个侧别最多保留几个检测。一个人只有一支球拍，所以同侧多出来的
    #: 必然是误检（常见于台面白边、手、头），按置信度留最高的那个。
    max_per_side: int = 1
    #: 一条轨迹要连续命中这么多帧才输出。用来滤掉"只闪一下"的假框；
    #: 1 表示关闭（训练/评估时保留全部检测）。
    confirm_hits: int = 1
    #: 是否输出"这一帧没看到球拍"的 visible=False 观测
    emit_lost_frames: bool = True
    tracker: TrackerConfig = field(default_factory=TrackerConfig)


class ProposalGate:
    """按 TrackNet/Rally 提议区间决定是否推理。"""

    def __init__(self, intervals: Sequence[tuple[float, float]] = (), *, expand_s: float = 0.0) -> None:
        self.intervals = sorted((float(t0) - expand_s, float(t1) + expand_s) for t0, t1 in intervals)

    def __bool__(self) -> bool:
        return bool(self.intervals)

    def contains(self, timestamp_s: float) -> bool:
        if not self.intervals:
            return True          # 没有提议区间时退化为全程推理
        return any(t0 <= timestamp_s <= t1 for t0, t1 in self.intervals)

    def total_duration_s(self) -> float:
        return sum(t1 - t0 for t0, t1 in self.intervals)


class RacketAdapter:
    def __init__(
        self,
        detector: Detector,
        config: AdapterConfig,
        *,
        log: ObservationLog | None = None,
        gate: ProposalGate | None = None,
    ) -> None:
        self.detector = detector
        self.config = config
        self.tracker = RacketTracker(config.tracker)
        self.log = log
        self.gate = gate or ProposalGate()
        self.frames_processed = 0
        self.frames_skipped = 0
        self.detections_used = 0
        self.dropped_low_conf = 0
        self.dropped_outside = 0
        self.dropped_same_side = 0
        self.pending_tracks = 0
        self._last_observations: list[RacketObservation] = []

    # ------------------------------------------------------------------ 单帧
    def process_frame(
        self,
        frame: Any,
        frame_id: int,
        timestamp_s: float,
        image_size: tuple[int, int],
    ) -> list[RacketObservation]:
        cfg = self.config
        self._last_observations = []

        if not self.gate.contains(timestamp_s):
            self.frames_skipped += 1
            return []

        raw = self.detector.detect(frame, image_size=image_size)
        detections = self._sanitize(raw, image_size)

        states = self.tracker.update(detections, frame_id, timestamp_s)
        observations: list[RacketObservation] = []
        for state in states:
            if state.status == "occluded":
                if cfg.emit_lost_frames:
                    observations.append(RacketObservation.lost(
                        session_id=cfg.session_id,
                        frame_id=frame_id,
                        timestamp_s=timestamp_s,
                        side=state.side,
                        track_id=state.track_id,
                        status="occluded",
                        source=cfg.source,
                        model_version=cfg.model_version,
                        image_size=image_size,
                    ))
                continue
            if not state.bbox:
                continue
            # 连续命中不足的轨迹先不输出：孤立的假框（空手、桌腿）通常只出现
            # 一两帧，真球拍会被持续跟踪。confirm_hits=1 表示不启用该过滤。
            if state.hits < cfg.confirm_hits:
                self.pending_tracks += 1
                continue
            observations.append(self._build(state, frame_id, timestamp_s, image_size))

        self.frames_processed += 1
        self._last_observations = observations
        if self.log is not None:
            for obs in observations:
                self.log.write(obs)
        return observations

    # ------------------------------------------------------------------ 批量
    def process_stream(
        self,
        frames: Iterable[tuple[Any, int, float, tuple[int, int]]],
    ) -> Iterator[list[RacketObservation]]:
        for frame, frame_id, timestamp_s, image_size in frames:
            yield self.process_frame(frame, frame_id, timestamp_s, image_size)

    # ------------------------------------------------------------------ 内部
    def _sanitize(self, raw: Sequence[Detection], image_size: tuple[int, int]) -> list[Detection]:
        cfg = self.config
        kept: list[Detection] = []
        for det in raw:
            if det.confidence < cfg.conf_thr:
                self.dropped_low_conf += 1
                continue
            clipped = clip_bbox(det.bbox, image_size[0], image_size[1])
            width = clipped[2] - clipped[0]
            height = clipped[3] - clipped[1]
            if width < 2 or height < 2:
                self.dropped_outside += 1
                continue
            side = det.side
            if side == "UNKNOWN" and det.class_id is not None:
                side = cfg.class_side_map.get(det.class_id, "UNKNOWN")
            if side == "UNKNOWN":
                side = side_from_x(
                    bbox_center(clipped)[0],
                    image_size[0],
                    net_x=cfg.net_x_ratio * image_size[0],
                    dead_zone_px=cfg.dead_zone_ratio * image_size[0],
                )
            kept.append(Detection(
                bbox=clipped,
                confidence=float(det.confidence),
                side=side,
                class_id=det.class_id,
                class_name=det.class_name,
            ))
        kept.sort(key=lambda d: -d.confidence)
        if cfg.max_per_side and cfg.max_per_side > 0:
            # 一个人只有一支球拍：同侧只留置信度最高的那个
            taken: dict[str, int] = {}
            deduped: list[Detection] = []
            for det in kept:
                count = taken.get(det.side, 0)
                if count >= cfg.max_per_side:
                    self.dropped_same_side += 1
                    continue
                taken[det.side] = count + 1
                deduped.append(det)
            kept = deduped
        kept = kept[: cfg.max_boxes]
        self.detections_used += len(kept)
        return kept

    def _build(
        self,
        state: Any,
        frame_id: int,
        timestamp_s: float,
        image_size: tuple[int, int],
    ) -> RacketObservation:
        cfg = self.config
        # 跟踪器已把框裁进图像；这里再过一遍契约校验，出问题宁可丢弃也不输出脏数据
        try:
            return RacketObservation(
                session_id=cfg.session_id,
                frame_id=frame_id,
                timestamp_s=timestamp_s,
                side=state.side if state.side in ("A", "B") else "UNKNOWN",
                bbox_xyxy=state.bbox,
                confidence=state.confidence,
                track_id=state.track_id,
                status="ok",
                visible=True,
                source=cfg.source,
                model_version=cfg.model_version,
                image_size=image_size,
            )
        except SchemaError:
            return RacketObservation.lost(
                session_id=cfg.session_id,
                frame_id=frame_id,
                timestamp_s=timestamp_s,
                side=state.side,
                track_id=state.track_id,
                status="lost",
                source=cfg.source,
                model_version=cfg.model_version,
                image_size=image_size,
            )

    # ------------------------------------------------------------------ 汇总
    def summary(self) -> dict[str, Any]:
        stats = self.tracker.stats()
        stats.update({
            "frames_processed": self.frames_processed,
            "frames_skipped": self.frames_skipped,
            "detections_used": self.detections_used,
            "dropped_low_conf": self.dropped_low_conf,
            "dropped_outside": self.dropped_outside,
            "dropped_same_side": self.dropped_same_side,
            "pending_tracks": self.pending_tracks,
            "source": self.config.source,
            "model_version": self.config.model_version,
        })
        return stats

    def close(self) -> None:
        if self.log is not None:
            self.log.close()
        try:
            self.detector.close()
        except Exception:  # 后端关闭失败不应该影响主流程
            pass
