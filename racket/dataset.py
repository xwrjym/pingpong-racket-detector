"""标注数据层：格式校验、按场次切分、导出 YOLO 数据集。

方案 §5 的硬要求在这里落实成代码：
* 每条标注至少有 frame_id / timestamp_s / bbox_xyxy / side / visible / 困难标签；
* 训练、验证、测试按**场次或视频**划分，同一回合的相邻帧不会被拆到两个集合。

数据格式（JSONL，一行一帧）::

    {"session_id":"s01","frame_id":900,"timestamp_s":15.0,
     "image_path":"D:/racket-data/frames/s01/000900.jpg",
     "boxes":[{"bbox_xyxy":[500,400,650,620],"side":"A","visible":true,
               "hard":["blur"]}]}

击球区间单独一个文件（一行一次击球）::

    {"session_id":"s01","hit_id":"h0042","t0_s":15.0,"t1_s":15.12,"side":"A"}
"""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .geometry import bbox_center, bbox_size
from .schema import SIDES, SchemaError, check_bbox

CLASS_NAMES: tuple[str, ...] = ("racket_A", "racket_B", "racket_unknown")
#: v0.1 默认单类别：侧别由几何判定，避免"有类无样本"导致模型在空类别上乱输出
CLASS_NAMES_SINGLE: tuple[str, ...] = ("racket",)
CLASS_IDS: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
HARD_TAGS: tuple[str, ...] = (
    "blur",        # 运动模糊
    "occluded",    # 被身体/球台遮挡
    "cross",       # 双人交叉
    "edge",        # 球拍半个出画
    "far",         # 远机位、小目标
    "low_light",   # 暗光
)


class DatasetError(ValueError):
    """标注数据不合法。"""


def class_name_for(side: str) -> str:
    return "racket_A" if side == "A" else "racket_B" if side == "B" else "racket_unknown"


@dataclass
class BoxLabel:
    bbox_xyxy: tuple[float, float, float, float]
    side: str = "UNKNOWN"
    visible: bool = True
    hard: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise DatasetError(f"side 必须是 {SIDES} 之一，收到 {self.side!r}")
        for tag in self.hard:
            if tag not in HARD_TAGS:
                raise DatasetError(f"未知困难标签 {tag!r}，可用：{HARD_TAGS}")
        if not self.visible and self.bbox_xyxy is not None:
            raise DatasetError("visible=False 的框必须为 null")

    @property
    def class_name(self) -> str:
        return class_name_for(self.side)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy else None,
            "side": self.side,
            "visible": self.visible,
            "hard": list(self.hard),
        }


@dataclass
class FrameLabel:
    session_id: str
    frame_id: int
    timestamp_s: float
    image_path: str
    boxes: list[BoxLabel] = field(default_factory=list)
    image_size: tuple[int, int] = (1920, 1080)
    #: 人工是否已经看过这一帧。False 表示"还没标"，**不能**当成"没有球拍"的负样本
    reviewed: bool = False

    def __post_init__(self) -> None:
        if not self.session_id:
            raise DatasetError("session_id 不能为空")
        if self.frame_id < 0:
            raise DatasetError("frame_id 不能为负")
        if self.timestamp_s < 0:
            raise DatasetError("timestamp_s 不能为负")
        for box in self.boxes:
            if box.visible:
                check_bbox(box.bbox_xyxy, self.image_size)

    @property
    def key(self) -> str:
        return f"{self.session_id}:{self.frame_id}"

    @property
    def visible_boxes(self) -> list[BoxLabel]:
        return [b for b in self.boxes if b.visible and b.side != "UNKNOWN"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "image_path": self.image_path,
            "image_size": list(self.image_size),
            "reviewed": self.reviewed,
            "boxes": [b.to_dict() for b in self.boxes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameLabel":
        size = tuple(data.get("image_size", (1920, 1080)))
        return cls(
            session_id=str(data["session_id"]),
            frame_id=int(data["frame_id"]),
            timestamp_s=float(data["timestamp_s"]),
            image_path=str(data.get("image_path", "")),
            boxes=[BoxLabel(
                bbox_xyxy=tuple(float(v) for v in b.get("bbox_xyxy")) if b.get("bbox_xyxy") else None,
                side=b.get("side", "UNKNOWN"),
                visible=bool(b.get("visible", True)),
                hard=tuple(b.get("hard", ())),
            ) for b in data.get("boxes", [])],
            image_size=(int(size[0]), int(size[1])),
            reviewed=bool(data.get("reviewed", False)),
        )


@dataclass
class HitLabel:
    session_id: str
    hit_id: str
    t0_s: float
    t1_s: float
    side: str = "UNKNOWN"

    def __post_init__(self) -> None:
        if self.t1_s < self.t0_s:
            raise DatasetError(f"击球区间非法：{self.t0_s} > {self.t1_s}")
        if self.side not in SIDES:
            raise DatasetError(f"side 必须是 {SIDES} 之一")

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "hit_id": self.hit_id,
            "t0_s": self.t0_s,
            "t1_s": self.t1_s,
            "side": self.side,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HitLabel":
        return cls(
            session_id=str(data["session_id"]),
            hit_id=str(data["hit_id"]),
            t0_s=float(data["t0_s"]),
            t1_s=float(data["t1_s"]),
            side=data.get("side", "UNKNOWN"),
        )


# --------------------------------------------------------------------- 读写
def load_frames(path: str | Path) -> list[FrameLabel]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"标注文件不存在：{path}")
    frames: list[FrameLabel] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                frame = FrameLabel.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, SchemaError, DatasetError) as exc:
                raise DatasetError(f"{path.name} 第 {lineno} 行不合法：{exc}") from exc
            if frame.key in seen:
                raise DatasetError(f"{path.name} 第 {lineno} 行重复：{frame.key}")
            seen.add(frame.key)
            frames.append(frame)
    return frames


def save_frames(frames: Iterable[FrameLabel], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for frame in frames:
            fh.write(json.dumps(frame.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def load_hits(path: str | Path) -> list[HitLabel]:
    path = Path(path)
    if not path.exists():
        return []
    hits: list[HitLabel] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                hits.append(HitLabel.from_dict(json.loads(line)))
    return hits


# --------------------------------------------------------------------- 统计
def dataset_stats(frames: Sequence[FrameLabel]) -> dict[str, Any]:
    sessions = sorted({f.session_id for f in frames})
    per_side = {"A": 0, "B": 0, "UNKNOWN": 0}
    hard: dict[str, int] = {}
    areas: list[float] = []
    for frame in frames:
        for box in frame.boxes:
            if not box.visible:
                continue
            per_side[box.side] += 1
            for tag in box.hard:
                hard[tag] = hard.get(tag, 0) + 1
            if box.bbox_xyxy:
                w, h = bbox_size(box.bbox_xyxy)
                areas.append(w * h)
    return {
        "frames": len(frames),
        "sessions": len(sessions),
        "session_ids": sessions,
        "boxes": sum(per_side.values()),
        "boxes_per_side": per_side,
        "hard_tags": hard,
        "frames_without_racket": sum(1 for f in frames if not f.visible_boxes),
        "median_box_area_px": sorted(areas)[len(areas) // 2] if areas else 0.0,
    }


def check_session_isolation(
    train: Sequence[FrameLabel], val: Sequence[FrameLabel], test: Sequence[FrameLabel]
) -> None:
    groups = {
        "train": {f.session_id for f in train},
        "val": {f.session_id for f in val},
        "test": {f.session_id for f in test},
    }
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = groups[a] & groups[b]
            if overlap:
                raise DatasetError(
                    f"{a} 与 {b} 共享场次 {sorted(overlap)}；"
                    "同一回合的相邻帧不能拆到不同集合（方案 §5）"
                )


# --------------------------------------------------------------------- 切分
def split_sessions(
    frames: Sequence[FrameLabel],
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[list[FrameLabel], list[FrameLabel], list[FrameLabel]]:
    """按场次切分，保证同场次的所有帧只落在一个集合里。"""
    sessions = sorted({f.session_id for f in frames})
    if len(sessions) < 3:
        raise DatasetError(
            f"只有 {len(sessions)} 个场次，无法按场次切出 train/val/test 三份；"
            "至少需要 3 段不同比赛或不同视频（方案 §5）"
        )
    rng = random.Random(seed)
    rng.shuffle(sessions)
    n = len(sessions)
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    if n - n_test - n_val < 1:
        n_val, n_test = 1, 1
    test_sessions = set(sessions[:n_test])
    val_sessions = set(sessions[n_test:n_test + n_val])
    train_sessions = set(sessions[n_test + n_val:])

    train = [f for f in frames if f.session_id in train_sessions]
    val = [f for f in frames if f.session_id in val_sessions]
    test = [f for f in frames if f.session_id in test_sessions]
    check_session_isolation(train, val, test)
    return train, val, test


# --------------------------------------------------------------------- 导出
def split_random(
    frames: Sequence[FrameLabel],
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[list[FrameLabel], list[FrameLabel], list[FrameLabel]]:
    """随机切分，让 train/val/test 都覆盖换边前后的画面。

    两种兜底切分各有用处，也各有代价：

    * ``split_timewise``：靠时间切，泄漏小，但**换边之后整段都会落在测试集**，
      模型从没见过两人互换位置的画面，测试指标会塌（实测 0.82 → 0.28）；
    * ``split_random``：按帧随机，覆盖两边分布，但相邻帧高度相关，指标偏乐观。

    单场视频无法同时满足"无泄漏"和"覆盖完整分布"，这就是方案 §5 坚持要
    多段视频的原因。两者都只用于止损检查点，不能写进最终报告当结论。
    """
    ordered = sorted(frames, key=lambda f: f.frame_id)
    rng = random.Random(seed)
    index = list(range(len(ordered)))
    rng.shuffle(index)
    n_test = max(1, round(len(index) * test_ratio))
    n_val = max(1, round(len(index) * val_ratio))
    test = [ordered[i] for i in index[:n_test]]
    val = [ordered[i] for i in index[n_test:n_test + n_val]]
    train = [ordered[i] for i in index[n_test + n_val:]]
    return train, val, test
def _yolo_lines(frame: FrameLabel, class_names: Sequence[str] = CLASS_NAMES) -> list[str]:
    lines: list[str] = []
    width, height = frame.image_size
    single = len(class_names) == 1
    for box in frame.boxes:
        if not box.visible or not box.bbox_xyxy:
            continue
        x1, y1, x2, y2 = box.bbox_xyxy
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        # 单类别模式下所有球拍都是 0 类，侧别交给几何判定（adapter.side_from_x）
        cls = 0 if single else CLASS_IDS[box.class_name]
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def split_cross_session(
    frames: Sequence[FrameLabel],
    val_ratio: float = 0.15,
    *,
    test_session: str | None = None,
) -> tuple[list[FrameLabel], list[FrameLabel], list[FrameLabel]]:
    """留一场做测试：train/val 只用其它场次，测试集是整段没见过的比赛。

    这是方案 §5 想要的评估方式——测试集和训练集出自不同的比赛、不同的选手，
    没有相邻帧泄漏，指标才可信。两个场次时：s01 训练（切 15% 做 val），s02 测试。
    """
    sessions = sorted({f.session_id for f in frames})
    if len(sessions) < 2:
        raise DatasetError("至少需要两个场次才能做跨场次评估")
    held_out = test_session or sessions[-1]
    if held_out not in sessions:
        raise DatasetError(f"测试场次 {held_out} 不在数据里：{sessions}")

    rest = [f for f in frames if f.session_id != held_out]
    ordered = sorted(rest, key=lambda f: (f.session_id, f.frame_id))
    n_val = max(1, round(len(ordered) * val_ratio))
    val = ordered[:n_val]
    train = ordered[n_val:]
    test = [f for f in frames if f.session_id == held_out]
    # train 与 val 同场次是设计如此（val 只是训练场次里切出来做早停的一段）；
    # 真正必须隔离的是测试场次，它不能和训练/验证有任何重叠。
    overlap = {f.session_id for f in test} & ({f.session_id for f in train}
                                              | {f.session_id for f in val})
    if overlap:
        raise DatasetError(f"测试场次与训练场次重叠：{sorted(overlap)}")
    return train, val, test


def split_timewise(
    frames: Sequence[FrameLabel],
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> tuple[list[FrameLabel], list[FrameLabel], list[FrameLabel]]:
    """按时间顺序切分，只在"场次不足 3 个"时兜底使用。

    **这违反方案 §5 的场次隔离要求**：相邻帧高度相关，指标会明显偏乐观。
    报告里必须标注，等拿到 3 段以上视频后换回 ``split_sessions``。
    """
    ordered = sorted(frames, key=lambda f: (f.session_id, f.timestamp_s))
    n = len(ordered)
    if n < 3:
        raise DatasetError("帧数太少，无法切分")
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    train = ordered[: n - n_val - n_test]
    val = ordered[n - n_val - n_test: n - n_test]
    test = ordered[n - n_test:]
    return train, val, test


def export_yolo(
    frames: Sequence[FrameLabel],
    dest: str | Path,
    *,
    splits: dict[str, Sequence[FrameLabel]],
    link_mode: str = "copy",
    class_names: Sequence[str] = CLASS_NAMES,
    backgrounds: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """导出 ultralytics 可直接训练的目录结构。

    ``dest/images/{split}`` + ``dest/labels/{split}`` + ``dest/data.yaml``。
    ``link_mode`` 支持 ``copy`` / ``hardlink``（后者省空间，需同盘）。
    传 ``class_names=("racket",)`` 可导出单类别数据集：侧别不靠模型分类，
    而由 ``adapter`` 按半台几何判定——v0.1 的默认做法。
    """
    if link_mode not in ("copy", "hardlink"):
        raise DatasetError("link_mode 只能是 copy 或 hardlink")
    if not class_names:
        raise DatasetError("class_names 不能为空")
    dest = Path(dest)
    counts: dict[str, int] = {}
    for split, items in splits.items():
        img_dir = dest / "images" / split
        lbl_dir = dest / "labels" / split
        # 必须清空：ultralytics 按目录扫描图片，残留的旧图（换了数据集之后）
        # 会被静默当成训练样本，指标和训练结果都会失真。
        for stale in (img_dir, lbl_dir):
            if stale.exists():
                shutil.rmtree(stale)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for frame in items:
            src = Path(frame.image_path)
            if not src.exists():
                raise DatasetError(f"帧文件不存在：{src}")
            if not frame.image_size or frame.image_size == (1920, 1080):
                pass  # 尺寸以标注为准；未标时沿用默认值
            stem = f"{frame.session_id}_{frame.frame_id:06d}"
            dst_img = img_dir / f"{stem}{src.suffix or '.jpg'}"
            if not dst_img.exists():
                if link_mode == "hardlink":
                    try:
                        os.link(src, dst_img)
                    except OSError:
                        shutil.copy2(src, dst_img)
                else:
                    shutil.copy2(src, dst_img)
            (lbl_dir / f"{stem}.txt").write_text(
                "\n".join(_yolo_lines(frame, class_names)) + ("\n" if frame.boxes else ""),
                encoding="utf-8",
            )
            written += 1
        counts[split] = written

    # 背景图（无标注文件内容）只进训练集：教模型"这些区域不是球拍"，
    # 对应头/耳朵/台面白边这类误检。
    if backgrounds:
        bg_dir = dest / "images" / "train"
        bg_lbl = dest / "labels" / "train"
        bg_dir.mkdir(parents=True, exist_ok=True)
        bg_lbl.mkdir(parents=True, exist_ok=True)
        added = 0
        for bg in backgrounds:
            src_bg = Path(bg)
            if not src_bg.exists():
                continue
            stem = f"bg_{src_bg.stem}"
            target = bg_dir / f"{stem}{src_bg.suffix or '.jpg'}"
            if not target.exists():
                shutil.copy2(src_bg, target)
            (bg_lbl / f"{stem}.txt").write_text("", encoding="utf-8")
            added += 1
        counts["train_backgrounds"] = added

    data_yaml = dest / "data.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    data_yaml.write_text(
        f"path: {dest.resolve().as_posix()}\n"
        # 空切分不写进 yaml，否则 ultralytics 会去扫一个空目录并报错
        + "".join(f"{split}: images/{split}\n" for split in splits if counts.get(split))
        + f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    return {"dest": str(dest), "counts": counts, "data_yaml": str(data_yaml)}
