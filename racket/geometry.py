"""框几何与侧别判定。

侧别约定沿用交接包 §3：``A`` = 物理球台左半台，``B`` = 右半台。
球拍属于哪一侧是"物理半台"，不是"哪个人"，所以换边之后同一个人会换 side，
这是设计意图，不是缺陷。
"""

from __future__ import annotations

from typing import Iterable, Sequence

BBox = tuple[float, float, float, float]


def bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_size(bbox: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (max(0.0, x2 - x1), max(0.0, y2 - y1))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """两个框的交并比；任一框退化时返回 0。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def clip_bbox(bbox: Sequence[float], width: int, height: int) -> BBox:
    """把框裁进图像内；完全出画时返回退化框（由调用方判 visible=False）。"""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    return (x1, y1, x2, y2)


def expand_bbox(bbox: Sequence[float], ratio: float) -> BBox:
    """按中心等比外扩，用于给 Proposal 区间留余量。"""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w, h = x2 - x1, y2 - y1
    dx, dy = w * ratio / 2.0, h * ratio / 2.0
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)


def side_from_x(
    center_x: float,
    image_width: int,
    *,
    net_x: float | None = None,
    dead_zone_px: float | None = None,
) -> str:
    """按球台中线的位置判 side。

    球拍压在中线上、或者两侧都在附近时，返回 ``UNKNOWN`` 而不是猜一个——
    方案第 3 节明确要求不猜。
    """
    if image_width <= 0:
        raise ValueError("image_width 必须为正")
    mid = image_width / 2.0 if net_x is None else float(net_x)
    zone = image_width * 0.02 if dead_zone_px is None else float(dead_zone_px)
    if abs(center_x - mid) <= zone:
        return "UNKNOWN"
    return "A" if center_x < mid else "B"


def side_from_bbox(
    bbox: Sequence[float],
    image_size: tuple[int, int],
    *,
    net_x: float | None = None,
    dead_zone_px: float | None = None,
) -> str:
    cx, _ = bbox_center(bbox)
    return side_from_x(cx, image_size[0], net_x=net_x, dead_zone_px=dead_zone_px)


def bbox_iou_matrix(
    left: Iterable[Sequence[float]], right: Iterable[Sequence[float]]
) -> list[list[float]]:
    """M×N 的 IoU 矩阵，供跟踪器做贪心匹配。"""
    left = list(left)
    right = list(right)
    return [[iou(a, b) for b in right] for a in left]
