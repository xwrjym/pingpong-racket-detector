"""用"同一侧出现两支球拍"来定位误检——一个人只有一支拍，多出来的必然是假的。

    python scripts/find_duplicate_side.py <weights> --fps 4

比按高度猜更可靠：不需要真值，也不依赖人头在画面哪个位置。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racket.detectors import YoloRacketDetector     # noqa: E402
from racket.detectors.base import DetectorConfig    # noqa: E402
from racket.geometry import iou, side_from_x        # noqa: E402


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="找同侧重复检测")
    parser.add_argument("weights")
    parser.add_argument("--video", default=r"C:\Users\rokmy\Downloads\test_4.mp4")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--out", default="reports/dup_side")
    parser.add_argument("--conf", type=float, default=0.25)
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = YoloRacketDetector(args.weights, DetectorConfig(conf_thr=args.conf,
                                                               imgsz=640, device="0"))

    suspicious = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            detections = detector.detect(frame, image_size=(width, height))
            groups: dict[str, list] = {"A": [], "B": []}
            for det in detections:
                side = side_from_x((det.bbox[0] + det.bbox[2]) / 2, width)
                if side in groups:
                    groups[side].append(det)
            for side, items in groups.items():
                if len(items) >= 2:
                    items.sort(key=lambda d: -d.confidence)
                    suspicious.append({
                        "frame_id": index,
                        "timestamp_s": round(index / src_fps, 2),
                        "side": side,
                        "boxes": [[round(v, 1) for v in d.bbox] for d in items],
                        "confs": [round(d.confidence, 3) for d in items],
                        "image": frame.copy(),
                    })
                    break
        index += 1
    cap.release()
    detector.close()

    print(f"扫描 {index} 帧，疑似重复 {len(suspicious)} 帧")
    tiles = []
    for item in suspicious[: args.limit]:
        frame = item["image"]
        # 取两个框的联合区域，放大 1.5 倍
        xs = [item["boxes"][0][0], item["boxes"][0][2], item["boxes"][1][0], item["boxes"][1][2]]
        ys = [item["boxes"][0][1], item["boxes"][0][3], item["boxes"][1][1], item["boxes"][1][3]]
        x1, x2 = max(0, min(xs) - 80), min(width, max(xs) + 80)
        y1, y2 = max(0, min(ys) - 80), min(height, max(ys) + 80)
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            continue
        tile = Image.fromarray(crop[:, :, ::-1]).resize((200, 200))
        draw = ImageDraw.Draw(tile)
        sx, sy = 200 / max(1, crop.shape[1]), 200 / max(1, crop.shape[0])
        for i, (box, conf) in enumerate(zip(item["boxes"], item["confs"])):
            color = (255, 60, 60) if i == 0 else (60, 160, 255)
            draw.rectangle([(box[0] - x1) * sx, (box[1] - y1) * sy,
                            (box[2] - x1) * sx, (box[3] - y1) * sy],
                           outline=color, width=2)
            draw.text((3, 3 + i * 12), f"{conf:.2f}", fill=color)
        draw.text((3, 190), f"t={item['timestamp_s']:.0f} {item['side']}", fill=(255, 255, 0))
        tiles.append(tile)

    if tiles:
        cols = 8
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 200, rows * 200), (20, 20, 20))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols) * 200, (i // cols) * 200))
        sheet.save(out / "duplicates.jpg", quality=88)
        print(f"拼图：{out / 'duplicates.jpg'}")

    (out / "duplicates.json").write_text(
        json.dumps([{k: v for k, v in s.items() if k != "image"} for s in suspicious],
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
