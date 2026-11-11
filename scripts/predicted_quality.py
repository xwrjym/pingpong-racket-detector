"""检查外推补位（status=predicted）的框到底准不准。

    python scripts/predicted_quality.py <weights> <labels> --imgsz 1280

外推能在数字上把覆盖率拉高，但如果补出来的框位置乱跑，就只是自欺。
这里统计：补位框有多少对得上真值、平均 IoU 是多少。
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image

    frames = [f for f in load_frames(args.labels) if f.reviewed]
    detector = YoloRacketDetector(args.weights,
                                  DetectorConfig(conf_thr=args.conf, imgsz=args.imgsz, device="0"))
    adapter = RacketAdapter(detector, AdapterConfig(session_id="s03", conf_thr=args.conf,
                                                   predict_missing=True))

    total = hit = 0
    ious: list[float] = []
    for frame in frames:
        image = np.array(Image.open(frame.image_path).convert("RGB"))[:, :, ::-1]
        obs = adapter.process_frame(image, frame.frame_id, frame.timestamp_s, frame.image_size)
        pred = [o for o in obs if o.status == "predicted" and o.bbox_xyxy]
        if not pred:
            continue
        truth = [b.bbox_xyxy for b in frame.visible_boxes]
        for o in pred:
            total += 1
            best = max((iou(o.bbox_xyxy, gt) for gt in truth), default=0.0)
            ious.append(best)
            hit += int(best >= 0.5)
    adapter.close()

    out = {
        "外推框数": total,
        "IoU≥0.5（位置基本准）": round(hit / total, 3) if total else None,
        "IoU中位数": round(sorted(ious)[len(ious) // 2], 3) if ious else None,
        "无对应真值（补错了）": round(1 - hit / total, 3) if total else None,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
