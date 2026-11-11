"""轻量球拍跟踪器。

不用 Kalman、不用 ByteTrack：球拍是低速大目标，快速运动时也只在 Proposal
区间里跑，IoU + 匀速预测已经够用，而且完全可解释、无额外依赖。

输出两条信息，正好对应方案第 3 节要求的两类证据：
* 每帧的「框 + 侧别 + track_id」
* 每支球拍的速度（px/s）与运动方向（度），用于击球方向相容性判断

「断轨」按方案 §10 的口径统计：同一条轨迹在存活期内连续丢失的时长，以及
因超过 ``max_age_frames`` 而被迫新建 id 的次数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .geometry import bbox_center, iou

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """检测器的原始输出（未跟踪、未做侧别纠错）。"""

    bbox: BBox
    confidence: float
    side: str = "UNKNOWN"
    class_id: int | None = None
    class_name: str | None = None


@dataclass
class TrackState:
    """跟踪器对某一帧某一支球拍的结论。"""

    track_id: str
    side: str
    frame_id: int
    timestamp_s: float
    status: str = "ok"                      # ok / occluded / lost
    bbox: BBox | None = None
    confidence: float | None = None
    velocity_px_s: tuple[float, float] = (0.0, 0.0)
    speed_px_s: float = 0.0
    direction_deg: float | None = None
    hits: int = 0
    age: int = 0
    silent_frames: int = 0
    predicted: bool = False

    @property
    def visible(self) -> bool:
        return self.status == "ok" and self.bbox is not None


@dataclass
class _Track:
    track_id: str
    side: str
    bbox: BBox
    last_center: tuple[float, float]
    last_timestamp_s: float
    last_frame_id: int
    hits: int = 1
    age: int = 0
    silent_frames: int = 0
    velocity: tuple[float, float] = (0.0, 0.0)
    gap_start_frame: int | None = None
    gap_frames: int = 0

    def predicted_bbox(self, dt: float) -> BBox:
        """匀速外推，用于快速挥拍时改善匹配。"""
        dx, dy = self.velocity[0] * dt, self.velocity[1] * dt
        x1, y1, x2, y2 = self.bbox
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


@dataclass
class TrackerConfig:
    iou_threshold: float = 0.3
    max_age_frames: int = 12
    #: 按时间判超龄（秒）。抽帧间隔大时比帧数更可靠，None 表示只看帧数
    max_age_s: float | None = None
    min_hits: int = 2
    velocity_smoothing: float = 0.5
    #: 低于这个速度时不判方向（像素噪声会让静止球拍的方向乱跳）
    min_speed_px_s: float = 20.0
    #: 门限：预测框与检测框中心距离超过「框对角线 × 该系数」不匹配
    gate_scale: float = 1.5
    #: 换边/交叉时是否允许 A、B 两侧的轨迹互换（默认不允许，宁可断轨）
    allow_side_switch: bool = False


class RacketTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._tracks: dict[str, _Track] = {}
        self._next_id: dict[str, int] = {}
        self._closed: list[dict[str, object]] = []
        self._gap_events: list[dict[str, object]] = []
        self.total_frames = 0

    # ------------------------------------------------------------------ 主循环
    def update(
        self,
        detections: Sequence[Detection],
        frame_id: int,
        timestamp_s: float,
    ) -> list[TrackState]:
        cfg = self.config
        self.total_frames += 1

        # 1) 已有轨迹全部进入"这一帧尚未匹配"状态
        for track in self._tracks.values():
            track.age += 1

        # 2) 贪心匹配：先按 IoU 降序，再按中心距离兜底
        pairs: list[tuple[float, str, int]] = []
        det_list = list(detections)
        for tid, track in self._tracks.items():
            dt = max(timestamp_s - track.last_timestamp_s, 1e-6)
            probe = track.predicted_bbox(dt) if track.hits >= 2 else track.bbox
            gate = math.hypot(probe[2] - probe[0], probe[3] - probe[1]) * cfg.gate_scale
            pc = bbox_center(probe)
            for idx, det in enumerate(det_list):
                if not cfg.allow_side_switch and track.side != "UNKNOWN" and det.side != "UNKNOWN" \
                        and track.side != det.side:
                    continue
                score = iou(probe, det.bbox)
                dc = bbox_center(det.bbox)
                dist = math.hypot(pc[0] - dc[0], pc[1] - dc[1])
                if score < cfg.iou_threshold:
                    # IoU 不够时，允许"中心几乎没动"的快速形变框
                    if dist > gate * 0.25:
                        continue
                    score = 0.0
                pairs.append((score - dist / max(gate, 1e-6) * 1e-3, tid, idx))
        pairs.sort(reverse=True)

        used_tracks: set[str] = set()
        used_dets: set[int] = set()
        matched: list[tuple[str, int, float]] = []
        for score, tid, idx in pairs:
            if tid in used_tracks or idx in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(idx)
            matched.append((tid, idx, score))

        # 3) 更新命中的轨迹
        states: list[TrackState] = []
        for tid, idx, _score in matched:
            track = self._tracks[tid]
            det = det_list[idx]
            dt = max(timestamp_s - track.last_timestamp_s, 1e-6)
            new_center = bbox_center(det.bbox)
            raw_v = (
                (new_center[0] - track.last_center[0]) / dt,
                (new_center[1] - track.last_center[1]) / dt,
            )
            a = cfg.velocity_smoothing if track.hits >= 2 else 1.0
            track.velocity = (
                a * raw_v[0] + (1 - a) * track.velocity[0],
                a * raw_v[1] + (1 - a) * track.velocity[1],
            )
            track.bbox = det.bbox
            track.last_center = new_center
            track.last_timestamp_s = timestamp_s
            track.last_frame_id = frame_id
            track.hits += 1
            track.silent_frames = 0
            if track.gap_start_frame is not None:
                self._gap_events.append({
                    "track_id": track.track_id,
                    "start_frame": track.gap_start_frame,
                    "end_frame": frame_id,
                    "frames": track.gap_frames,
                    # gap 之后才重新命中，dt 覆盖了 gap_frames+1 个间隔
                    "duration_s": round(dt * track.gap_frames / (track.gap_frames + 1), 4),
                })
                track.gap_start_frame = None
                track.gap_frames = 0
            if det.side != "UNKNOWN":
                track.side = det.side
            states.append(self._state(track, frame_id, timestamp_s, status="ok",
                                      bbox=det.bbox, confidence=det.confidence))

        # 4) 未命中的轨迹：超龄关闭，否则标记 occluded
        for tid in list(self._tracks):
            if tid in used_tracks:
                continue
            track = self._tracks[tid]
            track.silent_frames += 1
            if track.gap_start_frame is None:
                track.gap_start_frame = frame_id
            track.gap_frames += 1
            too_old = track.silent_frames > cfg.max_age_frames or (
                cfg.max_age_s is not None
                and (timestamp_s - track.last_timestamp_s) > cfg.max_age_s
            )
            if too_old:
                self._closed.append({
                    "track_id": track.track_id,
                    "first_frame": track.last_frame_id,
                    "last_frame": frame_id,
                    "hits": track.hits,
                    "reason": "lost",
                })
                if track.gap_start_frame is not None and track.gap_frames:
                    self._gap_events.append({
                        "track_id": track.track_id,
                        "start_frame": track.gap_start_frame,
                        "end_frame": frame_id,
                        "frames": track.gap_frames,
                        # 轨迹已断：用"最后一次命中到现在"的时长作为断轨时长
                        "duration_s": round(max(timestamp_s - track.last_timestamp_s, 0.0), 4),
                    })
                del self._tracks[tid]
            else:
                states.append(self._state(track, frame_id, timestamp_s, status="occluded"))

        # 5) 新建轨迹
        for idx, det in enumerate(det_list):
            if idx in used_dets:
                continue
            side = det.side if det.side in ("A", "B") else "U"
            self._next_id[side] = self._next_id.get(side, 0) + 1
            tid = f"racket_{side}_{self._next_id[side]:02d}"
            center = bbox_center(det.bbox)
            track = _Track(
                track_id=tid,
                side=det.side,
                bbox=det.bbox,
                last_center=center,
                last_timestamp_s=timestamp_s,
                last_frame_id=frame_id,
            )
            self._tracks[tid] = track
            states.append(self._state(track, frame_id, timestamp_s, status="ok",
                                      bbox=det.bbox, confidence=det.confidence))

        states.sort(key=lambda s: (s.side, s.track_id))
        return states

    # ------------------------------------------------------------------ 辅助
    def _state(
        self,
        track: _Track,
        frame_id: int,
        timestamp_s: float,
        *,
        status: str,
        bbox: BBox | None = None,
        confidence: float | None = None,
    ) -> TrackState:
        vx, vy = track.velocity
        speed = math.hypot(vx, vy)
        direction = None
        if speed >= self.config.min_speed_px_s:
            direction = round(math.degrees(math.atan2(vy, vx)), 2)
        return TrackState(
            track_id=track.track_id,
            side=track.side,
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            status=status,
            bbox=bbox,
            confidence=confidence,
            velocity_px_s=(round(vx, 3), round(vy, 3)),
            speed_px_s=round(speed, 3),
            direction_deg=direction,
            hits=track.hits,
            age=track.age,
            silent_frames=track.silent_frames,
        )

    def active_tracks(self) -> dict[str, str]:
        return {tid: t.side for tid, t in self._tracks.items()}

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id.clear()
        self._closed.clear()
        self._gap_events.clear()
        self.total_frames = 0

    def stats(self) -> dict[str, object]:
        """断轨与轨迹统计，直接对应方案 §10 的「球拍跟踪断轨时长」。"""
        gaps = [g for g in self._gap_events if g.get("frames")]
        gap_frames = sum(int(g["frames"]) for g in gaps)
        durations = [float(g["duration_s"]) for g in gaps if g.get("duration_s")]
        return {
            "tracks_created": sum(self._next_id.values()),
            "tracks_alive": len(self._tracks),
            "tracks_closed": len(self._closed),
            "gap_events": len(gaps),
            "gap_frames": gap_frames,
            "gap_ratio": round(gap_frames / max(self.total_frames, 1), 4),
            "max_gap_s": round(max(durations), 3) if durations else 0.0,
            "mean_gap_s": round(sum(durations) / len(durations), 3) if durations else 0.0,
            "id_switches": 0,   # 需要 GT 才能统计，见 eval.TrackMetrics
        }
