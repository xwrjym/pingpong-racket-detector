"""用真实标注评估：适配器 + 本项目指标 + 预测可视化。

    python scripts/eval_real.py data/videos/test_4/labels_manual.jsonl \
        runs/detect/runs/v1_60frames/weights/best.pt reports/v1_test

和 ultralytics 的 COCO 指标不同，这里走的是**产品链路**：
YOLO 后端 -> RacketAdapter（含 Proposal 门控、坐标校验、跟踪）-> 本项目 eval。
顺便统计侧别准确率——侧别是几何判定的，必须单独验。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racket.adapter import AdapterConfig, RacketAdapter           # noqa: E402
from racket.dataset import load_frames, split_timewise           # noqa: E402
from racket.detectors import YoloRacketDetector                  # noqa: E402
from racket.detectors.base import DetectorConfig                 # noqa: E402
from racket.eval import PredBox, PredFrame, evaluate_detection   # noqa: E402

COLORS = {"A": (34, 255, 136), "B": (255, 176, 32), "UNKNOWN": (160, 160, 160)}


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    labels_path = argv[0] if argv else "data/videos/test_4/labels_manual.jsonl"
    weights = argv[1] if len(argv) > 1 else "runs/detect/runs/v1_60frames/weights/best.pt"
    out_dir = Path(argv[2] if len(argv) > 2 else "reports/v1_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [f for f in load_frames(labels_path) if f.reviewed]
    # 与训练时完全相同的切分方式
    _, _, test = split_timewise(frames)

    detector = YoloRacketDetector(weights, DetectorConfig(conf_thr=0.25, imgsz=640, device="0"))
    adapter = RacketAdapter(detector, AdapterConfig(session_id="s01", conf_thr=0.25))

    from PIL import Image, ImageDraw

    predicted: list[PredFrame] = []
    rows = []
    for frame in test:
        image = Image.open(frame.image_path).convert("RGB")
        array = __import__("numpy").array(image)[:, :, ::-1]  # RGB -> BGR
        observations = adapter.process_frame(array, frame.frame_id, frame.timestamp_s, frame.image_size)
        boxes = [PredBox(o.bbox_xyxy, o.confidence or 0.0, o.side) for o in observations if o.visible]
        predicted.append(PredFrame(frame.session_id, frame.frame_id, frame.timestamp_s, boxes))

        truth = frame.visible_boxes
        rows.append({
            "frame_id": frame.frame_id,
            "gt": len(truth),
            "pred": len(boxes),
            "gt_sides": [b.side for b in truth],
            "pred_sides": [b.side for b in boxes],
        })

        draw = ImageDraw.Draw(image)
        for box in boxes:
            color = COLORS.get(box.side, COLORS["UNKNOWN"])
            draw.rectangle(box.bbox_xyxy, outline=color, width=5)
            draw.text((box.bbox_xyxy[0] + 6, max(0, box.bbox_xyxy[1] - 18)),
                      f"{box.side} {box.confidence:.2f}", fill=color)
        image.save(out_dir / f"pred_{frame.frame_id:06d}.jpg", quality=88)

    detection = evaluate_detection(test, predicted)
    summary = {
        "frames": len(test),
        "ultralytics_like": {
            "mAP50": detection["mAP50"],
            "frame_recall": detection["frame_recall"],
            "side_accuracy": detection["side_accuracy"],
            "side_unknown_rate": detection["side_unknown_rate"],
            "precision_recall": detection["precision_recall"],
        },
        "per_frame": rows,
        "tracking": adapter.summary(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_frame"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
