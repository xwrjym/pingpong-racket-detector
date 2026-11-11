"""在整段视频上搜误检候选：重点看头部高度带的预测。

    python scripts/mine_hard_negatives.py <weights> --fps 5 --out reports/head_fp

抽检 150 帧只有 2 个误检，但用户在实际视频里看到了"耳朵/头被认成球拍"，
说明误检集中在抽样没覆盖到的姿态。这里按更高帧率扫全片，把所有落在
头部高度带的检测裁出来，人工确认后作为**难负样本**加入训练。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racket.detectors import YoloRacketDetector     # noqa: E402
from racket.detectors.base import DetectorConfig    # noqa: E402


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="搜集头部高度带的误检候选")
    parser.add_argument("weights")
    parser.add_argument("--video", default=r"C:\Users\rokmy\Downloads\test_4.mp4")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--out", default="reports/head_fp")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--head-y", type=float, default=0.50,
                        help="中心点在这个高度比例以上就当成头部候选")
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args(argv)

    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    step = max(1, round(src_fps / args.fps))
    width = int(cap.get(cv2.CAP_PROP_WIDTH)) if False else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = YoloRacketDetector(args.weights, DetectorConfig(conf_thr=args.conf,
                                                               imgsz=640, device="0"))
    candidates = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            detections = detector.detect(frame, image_size=(width, height))
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                cy = (y1 + y2) / 2 / height
                if cy < args.head_y:
                    candidates.append({
                        "frame_id": index,
                        "timestamp_s": round(index / src_fps, 3),
                        "conf": round(det.confidence, 3),
                        "bbox_xyxy": [round(v, 1) for v in det.bbox],
                        "center_y_ratio": round(cy, 3),
                        "image": frame.copy(),
                    })
        index += 1
    cap.release()
    detector.close()

    print(f"扫描 {index} 帧，头部高度带候选 {len(candidates)} 个")
    by_frame: dict[int, dict] = {}
    for item in candidates:
        by_frame.setdefault(item["frame_id"], item)

    tiles = []
    for item in sorted(by_frame.values(), key=lambda c: -c["conf"])[: args.limit]:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        w, h = x2 - x1, y2 - y1
        pad_x, pad_y = w * 1.6, h * 1.6
        box = (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
               min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y)))
        crop = item["image"][box[1]:box[3], box[0]:box[2]]
        if crop.size == 0:
            continue
        tile = Image.fromarray(crop[:, :, ::-1]).resize((160, 160))
        draw = ImageDraw.Draw(tile)
        sx = 160 / max(1, crop.shape[1])
        sy = 160 / max(1, crop.shape[0])
        draw.rectangle([(x1 - box[0]) * sx, (y1 - box[1]) * sy,
                        (x2 - box[0]) * sx, (y2 - box[1]) * sy],
                       outline=(255, 60, 60), width=2)
        draw.text((3, 3), f"{item['conf']:.2f} t={item['timestamp_s']:.0f}", fill=(255, 255, 0))
        tiles.append(tile)

    if tiles:
        cols = 8
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 160, rows * 160), (20, 20, 20))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols) * 160, (i // cols) * 160))
        sheet.save(out / "head_candidates.jpg", quality=88)
        print(f"候选拼图：{out / 'head_candidates.jpg'}（{len(tiles)} 个）")

    (out / "candidates.json").write_text(
        json.dumps([{k: v for k, v in c.items() if k != "image"} for c in candidates],
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "candidate_frames.json").write_text(
        json.dumps(sorted(by_frame), ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
