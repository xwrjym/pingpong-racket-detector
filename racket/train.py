"""训练入口：标注 -> YOLO 数据集 -> 微调 -> 测试集评估 -> 导出 ONNX。

用法（装好 ultralytics 之后）::

    python -m racket.train --labels data/labels.jsonl --dest data/yolo \
        --epochs 120 --imgsz 640 --batch 8 --model yolov8n.pt

先在小样本上跑通再加量：方案 §12.2 的止损检查点只有 800–1200 帧。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

from .dataset import (
    CLASS_NAMES,
    CLASS_NAMES_SINGLE,
    DatasetError,
    FrameLabel,
    dataset_stats,
    export_yolo,
    load_frames,
    split_cross_session,
    split_random,
    split_sessions,
    split_timewise,
)


def build_yolo_dataset(
    labels_path: str | Path,
    dest: str | Path,
    *,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 0,
    link_mode: str = "copy",
    single_class: bool = True,
    allow_time_split: bool = True,
    split_mode: str = "time",
    background_dir: str | Path | None = None,
    test_session: str | None = None,
) -> dict[str, Any]:
    """准备 YOLO 数据集。

    ``single_class=True`` 时导出单类别 ``racket``：侧别不交给模型分类，而由
    ``adapter`` 按半台几何判定。标注里缺少某一侧样本时，多类别会退化成
    "空类别乱输出"，单类别没有这个问题。

    场次不足 3 个时自动退到时间切分，并在返回值里标 ``leakage=True``——
    这种切法会让指标偏乐观，报告里必须写明。
    """
    frames = load_frames(labels_path)
    total_frames = len(frames)
    # 只用人已复核过的帧。未复核 ≠ 负样本：把"还没标"当成"没有球拍"会让模型
    # 学会主动漏检。旧文件没有 reviewed 字段时退回全量，保持兼容。
    reviewed = [f for f in frames if f.reviewed]
    if reviewed:
        frames = reviewed
    stats = dataset_stats(frames)
    leakage = False
    if split_mode == "cross":
        # 显式要求跨场次时，必须优先于"按场次三分"：否则 3 个场次会被切成
        # 1 训练 + 1 验证 + 1 测试，训练数据只剩一场，和预期完全不同。
        train, val, test = split_cross_session(frames, val_ratio=0.15,
                                               test_session=test_session)
        leakage = False
    else:
        try:
            train, val, test = split_sessions(
                frames, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
            )
        except DatasetError:
            if not allow_time_split:
                raise
            if split_mode == "all":
                # 交付用：全部帧都进训练集，不再切测试集（也就没有可信指标）
                train, val, test = list(frames), list(frames[:min(20, len(frames))]), []
            elif split_mode == "random":
                train, val, test = split_random(frames, val_ratio=val_ratio, test_ratio=test_ratio)
            else:
                train, val, test = split_timewise(frames, val_ratio=val_ratio, test_ratio=test_ratio)
            leakage = True
    info = export_yolo(
        frames,
        dest,
        splits={"train": train, "val": val, "test": test},
        link_mode=link_mode,
        class_names=CLASS_NAMES_SINGLE if single_class else CLASS_NAMES,
        backgrounds=sorted(Path(background_dir).glob("*.jpg")) if background_dir else (),
    )
    info["stats"] = stats
    info["single_class"] = single_class
    info["leakage"] = leakage
    info["unreviewed_skipped"] = total_frames - len(frames)
    info["backgrounds"] = info["counts"].get("train_backgrounds", 0)
    info["split_frames"] = {"train": len(train), "val": len(val), "test": len(test)}
    info["split_sessions"] = {
        "train": sorted({f.session_id for f in train}),
        "val": sorted({f.session_id for f in val}),
        "test": sorted({f.session_id for f in test}),
    }
    return info


def train(
    data_yaml: str | Path,
    *,
    model: str = "yolov8n.pt",
    epochs: int = 120,
    imgsz: int = 640,
    batch: int = 8,
    device: str | int | None = None,
    project: str | Path = "runs",
    name: str = "racket_v0.1",
    patience: int = 30,
    seed: int = 0,
    resume: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """微调 YOLO。返回 best.pt 路径与训练耗时，便于落到评估报告。"""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("没有找到 ultralytics，请先运行 README 里的安装命令") from exc

    started = time.time()
    yolo = YOLO(model)
    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(project),
        "name": name,
        "patience": patience,
        "seed": seed,
        "exist_ok": True,
        "resume": resume,
        "plots": True,
        "val": True,
    }
    if device is not None:
        kwargs["device"] = device
    if extra:
        kwargs.update(extra)
    results = yolo.train(**kwargs)

    save_dir = Path(getattr(results, "save_dir", Path(project) / name))
    best = save_dir / "weights" / "best.pt"
    return {
        "best_weights": str(best),
        "save_dir": str(save_dir),
        "epochs": epochs,
        "imgsz": imgsz,
        "duration_s": round(time.time() - started, 1),
        "exists": best.exists(),
    }


def evaluate(weights: str | Path, data_yaml: str | Path, *, split: str = "test",
             imgsz: int = 640, device: str | int | None = None,
             workers: int = 0) -> dict[str, Any]:
    """在指定切分上跑 COCO 风格指标，并把关键的 AP/PR 抽成平铺字典。

    ``workers`` 默认 0：DSH 沙箱禁止多进程使用命名管道，非 0 会直接崩掉；
    单进程验证只是慢一点，结果一样。
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("没有找到 ultralytics") from exc
    model = YOLO(str(weights))
    kwargs: dict[str, Any] = {
        "data": str(data_yaml), "split": split, "imgsz": imgsz, "workers": workers,
    }
    if device is not None:
        kwargs["device"] = device
    metrics = model.val(**kwargs)
    summary: dict[str, Any] = {
        "split": split,
        "mAP50": float(getattr(metrics.box, "map50", 0.0)),
        "mAP50_95": float(getattr(metrics.box, "map", 0.0)),
        "precision": float(getattr(metrics.box, "mp", 0.0)),
        "recall": float(getattr(metrics.box, "mr", 0.0)),
    }
    names = getattr(model, "names", {}) or {}
    per_class = {}
    ap50 = getattr(metrics.box, "ap50", None)
    # numpy 数组不能直接做布尔判断（"or []" 会抛 ValueError）
    ap50_list = list(ap50) if ap50 is not None else []
    for i, ap in enumerate(ap50_list):
        per_class[str(names.get(i, i))] = float(ap)
    summary["ap50_per_class"] = per_class
    return summary


def export_onnx(weights: str | Path, *, imgsz: int = 640, opset: int = 12) -> str:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    return str(model.export(format="onnx", imgsz=imgsz, opset=opset))


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="球拍检测模型训练")
    parser.add_argument("--labels", type=Path, required=True, help="标注 JSONL")
    parser.add_argument("--dest", type=Path, default=Path("data/yolo"), help="YOLO 数据集输出目录")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="racket_v0.1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args(argv)

    info = build_yolo_dataset(
        args.labels, args.dest,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed,
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))

    result = train(
        info["data_yaml"],
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result["exists"]:
        metrics = evaluate(result["best_weights"], info["data_yaml"])
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        if args.export_onnx:
            print("ONNX:", export_onnx(result["best_weights"], imgsz=args.imgsz))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
