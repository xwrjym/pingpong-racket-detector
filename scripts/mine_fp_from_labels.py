"""用真值挖误检：凡是对不上任何标注框的检测，都是负样本。

    python scripts/mine_fp_from_labels.py <weights> data/videos/test_4/labels_manual.jsonl

和 ``mine_hard_negatives.py``（按"同侧重复"找）互补：
同侧重复抓不到"真球拍被漏检、只剩一个假框"的情况，用真值就能抓到。
裁出来的小图作为背景图加入训练，专门治"空手被认成球拍"。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racket.dataset import load_frames                      # noqa: E402
from racket.detectors import YoloRacketDetector             # noqa: E402
from racket.detectors.base import DetectorConfig            # noqa: E402
from racket.geometry import iou                             # noqa: E402


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="按真值挖误检并导出背景图")
    parser.add_argument("weights")
    parser.add_argument("labels")
    parser.add_argument("--out", default="data/hard_negatives_gt")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="用低阈值挖，把低置信的误检也收进来")
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--pad", type=float, default=0.35)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--max", type=int, default=400)
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image

    frames = [f for f in load_frames(args.labels) if f.reviewed]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.jpg"):
        stale.unlink()

    detector = YoloRacketDetector(args.weights, DetectorConfig(conf_thr=args.conf,
                                                               imgsz=640, device="0"))
    saved = 0
    total_fp = 0
    for frame in frames:
        if saved >= args.max:
            break
        image = Image.open(frame.image_path).convert("RGB")
        array = np.array(image)[:, :, ::-1]
        truth = [b.bbox_xyxy for b in frame.visible_boxes]
        for det in detector.detect(array, image_size=frame.image_size):
            if any(iou(gt, det.bbox) >= args.iou for gt in truth):
                continue
            total_fp += 1
            x1, y1, x2, y2 = det.bbox
            w, h = x2 - x1, y2 - y1
            pad_x, pad_y = w * args.pad, h * args.pad
            crop_box = (max(0, x1 - pad_x), max(0, y1 - pad_y),
                        min(frame.image_size[0], x2 + pad_x),
                        min(frame.image_size[1], y2 + pad_y))
            # 裁出来的块里不能有真球拍，否则会把真球拍教成背景
            if any(_overlap(crop_box, gt) > 0.02 for gt in truth):
                continue
            crop = image.crop(crop_box).resize((args.size, args.size))
            crop.save(out / f"fp_{saved:04d}.jpg", quality=92)
            saved += 1
    detector.close()

    print(json.dumps({
        "误检总数": total_fp,
        "导出背景图": saved,
        "输出目录": str(out),
        "conf": args.conf,
    }, ensure_ascii=False, indent=2))
    return 0


def _overlap(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = (b[2] - b[0]) * (b[3] - b[1])
    return inter / area if area > 0 else 0.0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
