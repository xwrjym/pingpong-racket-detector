"""切块推理（tiling）：把 1920 帧切成 2×2 小块分别推理，再合并。

    python scripts/tiled_eval.py <weights> <labels> --imgsz 960 --tiles 2

小目标召回低是因为整帧缩到 960 后球拍只剩二十几像素。切块后每块原分辨率
喂给模型，等效放大约 2 倍，且**不需要重训**。代价是推理次数翻 4 倍。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racket.dataset import load_frames                       # noqa: E402
from racket.detectors import YoloRacketDetector              # noqa: E402
from racket.detectors.base import DetectorConfig             # noqa: E402
from racket.geometry import iou                              # noqa: E402
from racket.tracker import Detection                         # noqa: E402


def nms(dets: list[Detection], thr: float = 0.5) -> list[Detection]:
    dets = sorted(dets, key=lambda d: -d.confidence)
    kept: list[Detection] = []
    for det in dets:
        if all(iou(det.bbox, k.bbox) < thr for k in kept):
            kept.append(det)
    return kept


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("weights")
    parser.add_argument("labels")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.15)
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image

    frames = [f for f in load_frames(args.labels) if f.reviewed]
    detector = YoloRacketDetector(args.weights,
                                  DetectorConfig(conf_thr=args.conf, imgsz=args.imgsz, device="0"))

    buckets = {"小 <50px": [0, 0], "中 50-90px": [0, 0], "大 ≥90px": [0, 0]}
    for frame in frames:
        image = np.array(Image.open(frame.image_path).convert("RGB"))[:, :, ::-1]
        h, w = image.shape[:2]
        n = args.tiles
        tile_w, tile_h = w // n, h // n
        ox, oy = int(tile_w * args.overlap), int(tile_h * args.overlap)
        dets: list[Detection] = []
        for ty in range(n):
            for tx in range(n):
                x0 = max(0, tx * tile_w - ox)
                y0 = max(0, ty * tile_h - oy)
                x1 = min(w, (tx + 1) * tile_w + ox)
                y1 = min(h, (ty + 1) * tile_h + oy)
                sub = image[y0:y1, x0:x1]
                for det in detector.detect(sub, image_size=(x1 - x0, y1 - y0)):
                    bx1, by1, bx2, by2 = det.bbox
                    dets.append(Detection(bbox=(bx1 + x0, by1 + y0, bx2 + x0, by2 + y0),
                                          confidence=det.confidence, side=det.side,
                                          class_id=det.class_id, class_name=det.class_name))
        dets = nms(dets)[:2]
        used: set[int] = set()
        for gt in frame.visible_boxes:
            gx1, gy1, gx2, gy2 = gt.bbox_xyxy
            width_px = gx2 - gx1
            key = "小 <50px" if width_px < 50 else ("中 50-90px" if width_px < 90 else "大 ≥90px")
            best, best_iou = -1, 0.0
            for i, det in enumerate(dets):
                if i in used:
                    continue
                value = iou(gt.bbox_xyxy, det.bbox)
                if value > best_iou:
                    best, best_iou = i, value
            hit = best >= 0 and best_iou >= 0.5
            if hit:
                used.add(best)
            buckets[key][0] += 1
            buckets[key][1] += int(hit)
    detector.close()

    out = {k: {"总数": v[0], "命中": v[1], "召回": round(v[1] / v[0], 3) if v[0] else 0.0}
           for k, v in buckets.items()}
    total = sum(v[0] for v in buckets.values())
    hits = sum(v[1] for v in buckets.values())
    out["全部"] = {"总数": total, "命中": hits, "召回": round(hits / total, 3) if total else 0.0}
    print(json.dumps({"tiles": args.tiles, "imgsz": args.imgsz, "conf": args.conf, **out},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
