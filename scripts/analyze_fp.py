"""分析误检：把没匹配上真值的预测裁出来，看看它们到底是什么。

    python scripts/analyze_fp.py <weights> --out reports/fp_analysis

输出：
* 误检数量、按画面高度分带的分布（人头一般在画面上半部）
* 一张误检裁剪拼图，直接用眼睛确认是耳朵、头还是别的东西
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="误检分析")
    parser.add_argument("weights")
    parser.add_argument("--labels", default="data/videos/test_4/labels_manual.jsonl")
    parser.add_argument("--out", default="reports/fp_analysis")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image, ImageDraw

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = [f for f in load_frames(args.labels) if f.reviewed]
    detector = YoloRacketDetector(args.weights, DetectorConfig(conf_thr=args.conf,
                                                               imgsz=640, device="0"))

    fps = []
    tp = 0
    for frame in frames:
        image = np.array(Image.open(frame.image_path).convert("RGB"))[:, :, ::-1]
        detections = detector.detect(image, image_size=frame.image_size)
        truth = [b.bbox_xyxy for b in frame.visible_boxes]
        used: set[int] = set()
        for det in detections:
            best, best_iou = -1, 0.0
            for i, gt in enumerate(truth):
                if i in used:
                    continue
                value = iou(gt, det.bbox)
                if value > best_iou:
                    best, best_iou = i, value
            if best >= 0 and best_iou >= args.iou:
                used.add(best)
                tp += 1
            else:
                x1, y1, x2, y2 = det.bbox
                fps.append({
                    "frame_id": frame.frame_id,
                    "conf": round(det.confidence, 3),
                    "center": (round((x1 + x2) / 2), round((y1 + y2) / 2)),
                    "size": (round(x2 - x1), round(y2 - y1)),
                    "y_ratio": round((y1 + y2) / 2 / frame.image_size[1], 3),
                    "crop": (frame.image_path, (x1, y1, x2, y2)),
                })
    detector.close()

    total = tp + len(fps)
    bands = {"上 1/3（头肩区）": 0, "中 1/3": 0, "下 1/3": 0}
    for item in fps:
        ratio = item["y_ratio"]
        key = "上 1/3（头肩区）" if ratio < 1 / 3 else "中 1/3" if ratio < 2 / 3 else "下 1/3"
        bands[key] += 1

    print(f"检查帧数 {len(frames)}，命中 {tp}，误检 {len(fps)}，"
          f"精确率 {tp / max(total, 1):.3f}")
    for key, value in bands.items():
        print(f"  {key}: {value}")

    # 误检裁剪拼图（带一圈上下文，方便辨认）
    tiles = []
    for item in fps[:48]:
        path, (x1, y1, x2, y2) = item["crop"]
        w, h = x2 - x1, y2 - y1
        pad_x, pad_y = w * 1.2, h * 1.2
        box = (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
               int(x2 + pad_x), int(y2 + pad_y))
        tile = Image.open(path).convert("RGB").crop(box).resize((160, 160))
        draw = ImageDraw.Draw(tile)
        # 把误检框画回去
        sx = 160 / max(1, box[2] - box[0])
        sy = 160 / max(1, box[3] - box[1])
        draw.rectangle([(x1 - box[0]) * sx, (y1 - box[1]) * sy,
                        (x2 - box[0]) * sx, (y2 - box[1]) * sy],
                       outline=(255, 60, 60), width=2)
        draw.text((3, 3), f"{item['conf']:.2f}", fill=(255, 255, 0))
        tiles.append(tile)

    if tiles:
        cols = 8
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 160, rows * 160), (20, 20, 20))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols) * 160, (i // cols) * 160))
        sheet_path = out / "false_positives.jpg"
        sheet.save(sheet_path, quality=88)
        print(f"误检拼图：{sheet_path}（{len(tiles)} 个）")

    import json
    (out / "false_positives.json").write_text(
        json.dumps([{k: v for k, v in f.items() if k != "crop"} for f in fps],
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
