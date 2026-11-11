"""球拍证据融合基线（方案 §7、§12.3、§12.4）。

这里只实现"给融合层用的打分函数"，不碰计分状态机、不改判罚规则。
两个原则写在代码里：

1. **距离不能单独定罪**：``racket_ball_relation`` 是时间与空间的乘积，
   任一项不满足就整体压低；方案 §12.1 明确要求联合多源证据。
2. **权重是待校准的实验基线**：``HitWeights`` 的数值只是起点，
   必须先在验证集上校准、在按场次隔离的测试集上冻结，不能当生产阈值用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .schema import HitCandidate

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class HitWeights:
    """实验基线权重，四项之和应约等于 1。"""

    trajectory_change: float = 0.35
    racket_ball_relation: float = 0.30
    ball_speed_change: float = 0.20
    racket_motion_direction: float = 0.15
    threshold: float = 0.60

    def total(self) -> float:
        return (self.trajectory_change + self.racket_ball_relation
                + self.ball_speed_change + self.racket_motion_direction)

    def normalized(self) -> "HitWeights":
        total = self.total()
        if total <= 0:
            raise ValueError("权重之和必须为正")
        return HitWeights(
            trajectory_change=self.trajectory_change / total,
            racket_ball_relation=self.racket_ball_relation / total,
            ball_speed_change=self.ball_speed_change / total,
            racket_motion_direction=self.racket_motion_direction / total,
            threshold=self.threshold,
        )


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _diag(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return math.hypot(x2 - x1, y2 - y1)


def temporal_factor(dt_s: float, max_dt_s: float = 0.20) -> float:
    """球拍观测与击球时刻的时间差，0 = 完全同时，1 = 同时。"""
    if max_dt_s <= 0:
        raise ValueError("max_dt_s 必须为正")
    return _clamp(1.0 - abs(dt_s) / max_dt_s)


def spatial_factor(racket_bbox: BBox, ball_center: tuple[float, float],
                   scale_ratio: float = 1.5, min_scale_px: float = 40.0) -> float:
    """球到球拍框的距离，按球拍自身尺寸归一化。

    用球拍框对角线做尺度基准，避免远近机位下同一物理距离被判成不同分。
    高斯形式让"擦边球"平滑衰减，而不是硬阈值跳变。
    """
    scale = max(_diag(racket_bbox) * scale_ratio, min_scale_px)
    cx = (racket_bbox[0] + racket_bbox[2]) / 2.0
    cy = (racket_bbox[1] + racket_bbox[3]) / 2.0
    dist = math.hypot(ball_center[0] - cx, ball_center[1] - cy)
    # 到框边缘的距离更符合物理直觉：球贴着框时距离为 0
    dx = max(racket_bbox[0] - ball_center[0], 0.0, ball_center[0] - racket_bbox[2])
    dy = max(racket_bbox[1] - ball_center[1], 0.0, ball_center[1] - racket_bbox[3])
    edge_dist = math.hypot(dx, dy)
    return _clamp(math.exp(-((min(dist, edge_dist * 2) / scale) ** 2)))


def racket_ball_relation(
    racket_bbox: BBox,
    ball_center: tuple[float, float],
    dt_s: float,
    *,
    max_dt_s: float = 0.20,
    scale_ratio: float = 1.5,
    temporal_weight: float = 0.5,
) -> float:
    """球拍与球的时空关系分数，取值 0~1。

    乘积形式：只要时间差过大或距离过远，整体就低。这比加权平均更难被
    "距离近但其实不是同一次击球"骗过去。
    """
    t = temporal_factor(dt_s, max_dt_s)
    s = spatial_factor(racket_bbox, ball_center, scale_ratio)
    # 几何平均再按 temporal_weight 向时间项倾斜
    combined = (t ** temporal_weight) * (s ** (1.0 - temporal_weight))
    return _clamp(combined)


def direction_compatibility(
    racket_velocity: tuple[float, float],
    ball_velocity_before: tuple[float, float],
    ball_velocity_after: tuple[float, float],
) -> float:
    """球拍运动方向与球速度变化方向的相容性，0~1。

    真实击球时，球的新方向大致沿球拍挥动方向；两者夹角越大分数越低。
    速度退化（静止）时返回 0，交由其它证据决定。
    """
    dvx = ball_velocity_after[0] - ball_velocity_before[0]
    dvy = ball_velocity_after[1] - ball_velocity_before[1]
    dv_norm = math.hypot(dvx, dvy)
    rv_norm = math.hypot(*racket_velocity)
    if dv_norm < 1e-6 or rv_norm < 1e-6:
        return 0.0
    cos = (racket_velocity[0] * dvx + racket_velocity[1] * dvy) / (rv_norm * dv_norm)
    return _clamp((cos + 1.0) / 2.0)


def score_hit(
    *,
    trajectory_change: float,
    racket_bbox: BBox | None,
    ball_center: tuple[float, float] | None,
    dt_s: float,
    ball_speed_change: float,
    racket_velocity: tuple[float, float] | None = None,
    ball_velocity_before: tuple[float, float] | None = None,
    ball_velocity_after: tuple[float, float] | None = None,
    weights: HitWeights | None = None,
) -> tuple[float, list[str]]:
    """计算 HIT 候选质量分数，返回 (score, evidence)。

    球拍不可用时自动退化成只有轨迹项与球速项的组合（方案 §12.4），
    分数会被系统性压低，调用方应据此降低自动判分覆盖率。
    """
    w = (weights or HitWeights()).normalized()
    evidence: list[str] = []

    trajectory = _clamp(trajectory_change)
    if trajectory > 0.5:
        evidence.append("trajectory_change")

    speed_change = _clamp(ball_speed_change)
    if speed_change > 0.5:
        evidence.append("ball_speed_change")

    relation = 0.0
    if racket_bbox is not None and ball_center is not None:
        relation = racket_ball_relation(racket_bbox, ball_center, dt_s)
        if relation > 0.5:
            evidence.append("racket_ball_close")

    direction = 0.0
    if racket_velocity is not None and ball_velocity_before is not None \
            and ball_velocity_after is not None:
        direction = direction_compatibility(racket_velocity, ball_velocity_before, ball_velocity_after)
        if direction > 0.6:
            evidence.append("racket_motion_aligned")

    if racket_bbox is None or ball_center is None:
        # 退化路径：把第二、四项的权重按比例分摊到剩下两项
        rest = w.trajectory_change + w.ball_speed_change
        if rest <= 0:
            return 0.0, ["racket_missing"]
        score = (w.trajectory_change * trajectory + w.ball_speed_change * speed_change) / rest
        evidence.append("degraded_no_racket")
        return _clamp(score), evidence

    score = (
        w.trajectory_change * trajectory
        + w.racket_ball_relation * relation
        + w.ball_speed_change * speed_change
        + w.racket_motion_direction * direction
    )
    return _clamp(score), evidence


def is_suspect_hit(
    *,
    hit_y_px: float | None,
    table_top_y_px: float | None,
    table_bottom_y_px: float | None,
    bounce_times_s: Sequence[float],
    hit_timestamp_s: float,
    bounce_window_s: float = 0.15,
) -> bool:
    """方案 §12.3 的低风险诊断标记。

    当 HIT 候选落在台面高度带、且与落台候选时间接近时，标 ``suspect_hit``。
    **默认不否决 HIT**：真实击球后的球也可能短暂进入同一高度带。
    这是影子证据与数据筛选工具，不是判罚规则。
    """
    if hit_y_px is None or table_top_y_px is None or table_bottom_y_px is None:
        return False
    if not (table_top_y_px <= hit_y_px <= table_bottom_y_px):
        return False
    return any(abs(hit_timestamp_s - t) <= bounce_window_s for t in bounce_times_s)


def build_candidate(
    *,
    session_id: str,
    sequence: int,
    timestamp_s: float,
    side: str,
    score: float,
    evidence: Sequence[str],
    sources: Sequence[str] = ("racket_model_a",),
    weights: HitWeights | None = None,
) -> HitCandidate:
    """把打分结果封成方案 §7 的 hit_candidate 结构。"""
    w = weights or HitWeights()
    decision = "candidate" if score >= w.threshold else "pending"
    return HitCandidate(
        event_id=f"{session_id}:hit:{sequence:04d}",
        timestamp_s=timestamp_s,
        side=side if side in ("A", "B") else "UNKNOWN",
        sources=tuple(sources),
        evidence=tuple(evidence),
        quality_score=score,
        decision=decision,
    )
