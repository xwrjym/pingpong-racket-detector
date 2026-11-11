"""检查"标注是不是模型的翻版"：计算每个 GT 框与最佳预测的 IoU 分布。

    python scripts/label_independence.py <weights> <labels> --imgsz 1280

如果 IoU 中位数接近 0.95，说明标注基本照抄了模型输出——用这个模型去评这份标注
就是在自证，指标会虚高。这是半自动标注必须防的坑。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racket.adapter import AdapterConfig, RacketAdapter      # noqa: E402
from racket.dataset import load_frames                       # noqa: E402
from racket.detectors import YoloRacketDetector              # noqa: E402
from racket.detectors.base import DetectorConfig             # noqa: E402
from racket.geometry import iou                              # noqa: E402


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("weights")
    parser.add_argument("labels")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image

    frames = [f for f in load_frames(args.labels) if f.reviewed]
    detector = YoloRacketDetector(args.weights,
                                  DetectorConfig(conf_thr=args.conf, imgsz=args.imgsz, device="0"))
    adapter = RacketAdapter(detector, AdapterConfig(session_id="s04", conf_thr=args.conf))

    best_ious: list[float] = []
    for frame in frames:
        image = np.array(Image.open(frame.image_path).convert("RGB"))[:, :, ::-1]
        obs = adapter.process_frame(image, frame.frame_id, frame.timestamp_s, frame.image_size)
        preds = [o.bbox_xyxy for o in obs if o.visible and o.bbox_xyxy]
        used: set[int] = set()
        for gt in frame.visible_boxes:
            best, best_iou = -1, 0.0
            for i, box in enumerate(preds):
                if i in used:
                    continue
                value = iou(gt.bbox_xyxy, box)
                if value > best_iou:
                    best, best_iou = i, value
            if best >= 0 and best_iou >= 0.5:
                used.add(best)
            best_ious.append(best_iou)
    detector.close()

    arr = sorted(best_ious)
    n = len(arr)
    out = {
        "框数": n,
        "IoU中位数": round(statistics.median(arr), 4) if n else None,
        "IoU≥0.9占比": round(sum(1 for v in arr if v >= 0.9) / n, 3) if n else None,
        "IoU≥0.75占比": round(sum(1 for v in arr if v >= 0.75) / n, 3) if n else None,
        "IoU<0.5占比": round(sum(1 for v in arr if v < 0.5) / n, 3) if n else None,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
