"""半自动预标注：先用零训练方法粗筛球拍位置，人工只做修正。

思路（不需要任何权重）：
1. 帧间差分拿到"动的东西"，球拍挥动时必然出现在这里；
2. 在运动区域附近的边缘图上做圆检测——球拍头是整幅画面里少见的近圆形物体；
3. 圆越大、边缘越完整、运动越明显，分数越高；
4. 按分数排序、去重，交给人工在标注工具里确认或拖框。

这套方法召回有限（球拍静止、严重运动模糊时会漏），它的定位是**减少手标量**，
不是替代检测模型。方案 §12.2 的止损检查点仍然要用真人修正后的框来算指标。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .dataset import FrameLabel, BoxLabel, save_frames
from .geometry import iou, side_from_x


@dataclass
class PrelabelConfig:
    #: 处理时的缩放宽度；越小越快，越小也越容易漏小目标
    work_width: int = 960
    #: 球拍头半径范围，按画面宽度比例给（业余机位下实测大约这个区间）
    min_radius_ratio: float = 0.015
    max_radius_ratio: float = 0.070
    #: 帧差阈值（0~255），太低会把噪声当运动
    motion_thr: float = 22.0
    #: 圆检测的累加器阈值（0~1，越大越严格）
    hough_threshold: float = 0.28
    #: 候选框半边长 = 半径 × 该比例；必须和人工标注习惯一致，否则白算 IoU
    box_pad_ratio: float = 1.6
    #: 半径搜索步长（像素），越细定位越准、越慢
    radius_step: int = 2
    #: Canny 的 sigma
    edge_sigma: float = 1.3
    #: 只保留这么多候选，避免人工被噪声框淹没
    max_candidates: int = 3
    #: 低于该分数的候选直接丢弃
    min_score: float = 0.30
    #: 圆内肤色像素占比超过该值就判为人头/手臂，直接丢弃
    max_skin_fraction: float = 0.30
    net_x_ratio: float = 0.5
    dead_zone_ratio: float = 0.02
    #: 忽略画面底部（比分条/水印区域）
    ignore_bottom_ratio: float = 0.06


@dataclass
class Candidate:
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    radius: float
    side: str = "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": [round(v, 2) for v in self.bbox_xyxy],
            "score": round(self.score, 4),
            "radius": round(self.radius, 2),
            "side": self.side,
        }


def _load_rgb(path: str | Path, width: int) -> np.ndarray:
    """读成 RGB 浮点数组（保留颜色：球拍胶皮的颜色是最好的先验之一）。"""
    from PIL import Image

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        if rgb.width > width:
            ratio = width / rgb.width
            rgb = rgb.resize((width, max(1, int(round(rgb.height * ratio)))), Image.BILINEAR)
        return np.asarray(rgb, dtype=np.float32)


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return rgb.mean(axis=2)


def _load_gray(path: str | Path, width: int) -> np.ndarray:
    """兼容旧调用：只给灰度。"""
    return _to_gray(_load_rgb(path, width))


def _skin_fraction(rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    """人脸/手臂的经典阈值判定。球拍几乎不会是肤色，头却总是圆的。"""
    patch = rgb[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    r, g, b = patch[..., 0], patch[..., 1], patch[..., 2]
    mx = patch.max(axis=2)
    mn = patch.min(axis=2)
    cond = (r > 95) & (g > 40) & (b > 20) & ((mx - mn) > 15) & (np.abs(r - g) > 15) & (r > g) & (r > b)
    return float(cond.mean())


def _color_prior(rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    """红/蓝等高饱和橡胶，或很暗的黑色胶皮，比随机圆物更像球拍。"""
    patch = rgb[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    mx = patch.max(axis=2) / 255.0
    mn = patch.min(axis=2) / 255.0
    sat = (mx - mn) / np.clip(mx, 1e-6, None)
    colorful = float(((sat > 0.45) & (mx > 0.25)).mean())
    dark = float((mx < 0.35).mean())
    return float(np.clip(colorful * 0.8 + dark * 0.4, 0.0, 1.0))


def _motion_mask(current: np.ndarray, previous: np.ndarray | None, thr: float) -> np.ndarray:
    if previous is None or previous.shape != current.shape:
        return np.ones_like(current, dtype=bool)
    diff = np.abs(current - previous)
    return diff > thr


def _circle_candidates(
    rgb: np.ndarray,
    motion: np.ndarray,
    config: PrelabelConfig,
) -> list[Candidate]:
    """在运动掩膜附近的边缘图上做圆检测。"""
    from scipy.ndimage import binary_dilation, maximum_filter
    from skimage.feature import canny
    from skimage.transform import hough_circle, hough_circle_peaks

    height, width = rgb.shape[:2]
    gray = _to_gray(rgb)
    radii_min = max(4, int(round(width * config.min_radius_ratio)))
    radii_max = max(radii_min + 2, int(round(width * config.max_radius_ratio)))
    radii = np.arange(radii_min, radii_max + 1, max(1, config.radius_step))

    blurred = maximum_filter(gray, size=3)
    edges = canny(blurred / 255.0 if blurred.max() > 1.5 else blurred, sigma=config.edge_sigma)
    # 运动区域外扩，容忍球拍静止的球头部分
    motion_region = binary_dilation(motion, iterations=max(3, width // 120))

    # 峰值太密时会互相抑制，把真正的球拍顶掉，因此拉开最小间距
    min_distance = max(12, width // 32)
    accum, cx, cy, radius = hough_circle_peaks(
        hough_circle(edges, radii),
        radii,
        total_num_peaks=max(8, config.max_candidates * 4),
        threshold=config.hough_threshold,
        normalize=True,
        min_xdistance=min_distance,
        min_ydistance=min_distance,
    )
    if len(accum) == 0:
        return []

    ignore_bottom = int(height * (1.0 - config.ignore_bottom_ratio))
    candidates: list[Candidate] = []
    max_accum = float(accum.max()) if accum.max() > 0 else 1.0
    for score_raw, x, y, r in zip(accum, cx, cy, radius):
        if y > ignore_bottom or r <= 1:
            continue
        x0, x1 = max(int(x - r), 0), min(int(x + r) + 1, width)
        y0, y1 = max(int(y - r), 0), min(int(y + r) + 1, height)
        if x1 <= x0 or y1 <= y0:
            continue
        motion_frac = float(motion_region[y0:y1, x0:x1].mean())
        skin = _skin_fraction(rgb, x0, y0, x1, y1)
        if skin > config.max_skin_fraction:
            continue                      # 圆形的头/肩膀，不是球拍
        color = _color_prior(rgb, x0, y0, x1, y1)
        score = (0.45 * (float(score_raw) / max_accum)
                 + 0.30 * motion_frac
                 + 0.25 * color)
        if score < config.min_score:
            continue
        pad = r * config.box_pad_ratio
        candidates.append(Candidate(
            bbox_xyxy=(float(x - pad), float(y - pad), float(x + pad), float(y + pad)),
            score=score,
            radius=float(r),
            side=side_from_x(float(x), width,
                             net_x=config.net_x_ratio * width,
                             dead_zone_px=config.dead_zone_ratio * width),
        ))

    # 分数降序 + IoU 去重
    candidates.sort(key=lambda c: -c.score)
    kept: list[Candidate] = []
    for cand in candidates:
        if all(iou(cand.bbox_xyxy, other.bbox_xyxy) < 0.45 for other in kept):
            kept.append(cand)
        if len(kept) >= config.max_candidates:
            break
    return kept


def prelabel_frames(
    items: Sequence[dict[str, Any]],
    config: PrelabelConfig | None = None,
    *,
    progress_every: int = 50,
) -> tuple[list[FrameLabel], list[dict[str, Any]]]:
    """``items`` 每项至少含 image_path；可选 session_id / frame_id / timestamp_s。"""
    cfg = config or PrelabelConfig()
    frames: list[FrameLabel] = []
    meta: list[dict[str, Any]] = []
    previous_gray: np.ndarray | None = None
    for index, item in enumerate(items):
        path = Path(item["image_path"])
        width = item.get("image_width") or cfg.work_width
        rgb = _load_rgb(path, cfg.work_width)
        gray = _to_gray(rgb)
        motion = _motion_mask(gray, previous_gray, cfg.motion_thr)
        previous_gray = gray

        candidates = _circle_candidates(rgb, motion, cfg)
        image_w = int(item.get("image_width", 1920))
        image_h = int(item.get("image_height", 1080))
        scale = image_w / gray.shape[1]
        boxes = []
        for cand in candidates:
            x1, y1, x2, y2 = (v * scale for v in cand.bbox_xyxy)
            # 检测候选可能压到画外，标注必须裁进图像，否则契约校验会拒绝
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(image_w), x2), min(float(image_h), y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            boxes.append(BoxLabel(
                bbox_xyxy=(x1, y1, x2, y2),
                side=cand.side,
                visible=True,
                hard=(("blur",) if cand.score < 0.4 else ()),
            ))
        frame = FrameLabel(
            session_id=str(item.get("session_id", "s01")),
            frame_id=int(item.get("frame_id", index)),
            timestamp_s=float(item.get("timestamp_s", index / 30.0)),
            image_path=str(path),
            boxes=boxes,
            image_size=(int(item.get("image_width", 1920)), int(item.get("image_height", 1080))),
        )
        frames.append(frame)
        meta.append({
            "session_id": frame.session_id,
            "frame_id": frame.frame_id,
            "candidates": [c.to_dict() for c in candidates],
        })
        if progress_every and (index + 1) % progress_every == 0:
            print(f"  预标注 {index + 1}/{len(items)} 帧，最近一帧候选 {len(candidates)} 个")
    return frames, meta


def prelabel_manifest(
    manifest: str | Path,
    labels_out: str | Path,
    meta_out: str | Path | None = None,
    config: PrelabelConfig | None = None,
) -> dict[str, Any]:
    """读 manifest（JSONL，一行一帧）-> 写草稿标注 + 候选元数据。"""
    items: list[dict[str, Any]] = []
    with open(manifest, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    frames, meta = prelabel_frames(items, config)
    save_frames(frames, labels_out)
    if meta_out:
        with open(meta_out, "w", encoding="utf-8") as fh:
            for record in meta:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "frames": len(frames),
        "frames_with_candidates": sum(1 for m in meta if m["candidates"]),
        "labels": str(labels_out),
        "meta": str(meta_out) if meta_out else None,
    }


def build_manifest(
    frames_dir: str | Path,
    manifest_out: str | Path,
    *,
    session_id: str = "s01",
    fps: float = 30.0,
    image_width: int = 1920,
    image_height: int = 1080,
) -> dict[str, Any]:
    """把抽好的帧目录做成 manifest；文件名里的数字就是 frame_id。"""
    frames_dir = Path(frames_dir)
    images = sorted(
        (p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")),
        key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0),
    )
    if not images:
        raise FileNotFoundError(f"{frames_dir} 里没有图片")
    manifest_out = Path(manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with manifest_out.open("w", encoding="utf-8") as fh:
        for index, path in enumerate(images):
            digits = "".join(ch for ch in path.stem if ch.isdigit())
            frame_id = int(digits) if digits else index
            fh.write(json.dumps({
                "session_id": session_id,
                "frame_id": frame_id,
                "timestamp_s": round(frame_id / fps, 6),
                "image_path": str(path.resolve()),
                "image_width": image_width,
                "image_height": image_height,
            }, ensure_ascii=False) + "\n")
    return {"images": len(images), "manifest": str(manifest_out), "session_id": session_id}
