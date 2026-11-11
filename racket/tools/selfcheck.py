"""全链路自检：不需要真机视频、不需要 ultralytics，验证工程是可运行的。

    python -m racket.tools.selfcheck

做四件事：
1. 生成合成画面与 GT 标注；
2. 跑半自动预标注，和 GT 比对（证明预标注不是随机框）；
3. 导出 YOLO 数据集（证明训练数据格式没问题）；
4. 用 GT 框走一遍适配器 + 评估 + 报告（证明指标链路是通的）。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

from ..adapter import AdapterConfig, RacketAdapter
from ..dataset import HitLabel, export_yolo, load_frames, split_sessions
from ..eval import PredBox, PredFrame, evaluate_detection, evaluate_hit_windows, format_report
from ..geometry import iou
from ..prelabel import PrelabelConfig, prelabel_manifest
from ..tracker import Detection
from .synthetic import generate


class _GtDetector:
    """把 GT 框当检测结果：用来单独验证"适配器 -> 评估"这一段没有逻辑错误。"""

    def __init__(self) -> None:
        self.boxes: list[Any] = []

    def detect(self, frame: Any, *, image_size: tuple[int, int] | None = None) -> list[Detection]:
        return [Detection(bbox=b.bbox_xyxy, confidence=0.95, side=b.side) for b in self.boxes]

    def close(self) -> None:
        return None


def run(workdir: str | Path | None = None, *, keep: bool = False) -> dict[str, Any]:
    tmp = None
    if workdir is None:
        tmp = tempfile.TemporaryDirectory()
        workdir = Path(tmp.name)
    workdir = Path(workdir)
    report: dict[str, Any] = {}

    info = generate(workdir / "synthetic", sessions=3, frames_per_session=6)
    report["synthetic"] = info

    report["prelabel"] = prelabel_manifest(
        info["manifest"],
        workdir / "prelabel.jsonl",
        workdir / "prelabel_meta.jsonl",
        PrelabelConfig(max_candidates=2, min_score=0.2),
    )

    gt_frames = load_frames(info["gt_labels"])
    draft_frames = load_frames(workdir / "prelabel.jsonl")

    gt_by_key = {(f.session_id, f.frame_id): f for f in gt_frames}
    hits = side_ok = 0
    ious: list[float] = []
    for draft in draft_frames:
        gt = gt_by_key.get((draft.session_id, draft.frame_id))
        if not gt or not gt.visible_boxes:
            continue
        gt_box = gt.visible_boxes[0]
        best_iou, best_side = 0.0, None
        for cand in draft.visible_boxes:
            value = iou(gt_box.bbox_xyxy, cand.bbox_xyxy)
            if value > best_iou:
                best_iou, best_side = value, cand.side
        ious.append(best_iou)
        if best_iou >= 0.5:
            hits += 1
            side_ok += int(best_side == gt_box.side)
    report["prelabel_quality"] = {
        "frames": len(ious),
        "iou50_rate": round(hits / len(ious), 4) if ious else 0.0,
        "mean_best_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
        "side_accuracy": round(side_ok / hits, 4) if hits else 0.0,
        "note": "合成画面的自检数字，不能当作真实精度",
    }

    train, val, test = split_sessions(gt_frames, seed=1)
    export_info = export_yolo(gt_frames, workdir / "yolo",
                              splits={"train": train, "val": val, "test": test})
    report["yolo_export"] = export_info["counts"]

    detector = _GtDetector()
    adapter = RacketAdapter(detector, AdapterConfig(session_id="s01"))
    predicted: list[PredFrame] = []
    current_session = None
    for frame in gt_frames:
        if frame.session_id != current_session:
            adapter.tracker.reset()          # 换场次必须重置跟踪，避免跨场串轨
            adapter.config.session_id = frame.session_id
            current_session = frame.session_id
        detector.boxes = frame.visible_boxes
        obs = adapter.process_frame(None, frame.frame_id, frame.timestamp_s, frame.image_size)
        predicted.append(PredFrame(
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            timestamp_s=frame.timestamp_s,
            boxes=[PredBox(o.bbox_xyxy, o.confidence or 0.0, o.side) for o in obs if o.visible],
        ))

    detection = evaluate_detection(gt_frames, predicted)
    report["gt_driven_eval"] = detection
    report["tracking"] = adapter.summary()

    first = gt_frames[0]
    hits_eval = evaluate_hit_windows(
        [HitLabel(session_id=first.session_id, hit_id="h1",
                  t0_s=first.timestamp_s, t1_s=first.timestamp_s + 0.05,
                  side=first.visible_boxes[0].side)],
        predicted,
    )
    report["hit_window"] = hits_eval
    report["markdown"] = format_report(detection, hits=hits_eval, tracking=adapter.summary())

    if tmp is not None and not keep:
        tmp.cleanup()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):   # Windows 控制台默认 GBK，中文会乱码
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="全链路自检")
    parser.add_argument("--out", type=Path, default=None, help="把 Markdown 自检报告写到该路径")
    parser.add_argument("--keep", action="store_true", help="保留中间产物")
    parser.add_argument("--workdir", type=Path, default=None)
    args = parser.parse_args(argv)

    result = run(workdir=args.workdir, keep=args.keep)
    print(json.dumps({k: v for k, v in result.items() if k != "markdown"},
                     ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result["markdown"], encoding="utf-8")
        print(f"\n报告已写入 {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
