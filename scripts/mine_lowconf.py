"""扫描视频里的低置信度检测——"空手被认成球拍"基本都藏在这一档。

    python scripts/mine_lowconf.py <weights> --video ... --conf-min 0.15 --conf-max 0.60

输出拼图供人工判定；确认没有球拍的那些框，再用 --export 导出成背景图。
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
    parser = argparse.ArgumentParser(description="低置信度检测挖掘")
    parser.add_argument("weights")
    parser.add_argument("--video", default=r"C:\Users\rokmy\Downloads\test_2.mp4")
    parser.add_argument("--out", default="reports/lowconf")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--conf-min", type=float, default=0.15)
    parser.add_argument("--conf-max", type=float, default=0.60)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--pad", type=float, default=0.6, help="裁剪外扩比例")
    parser.add_argument("--size", type=int, default=200)
    args = parser.parse_args(argv)

    import cv2
    from PIL import Image, ImageDraw

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    step = max(1, round(fps / args.fps))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = YoloRacketDetector(args.weights,
                                  DetectorConfig(conf_thr=args.conf_min, imgsz=640, device="0"))

    found = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            for det in detector.detect(frame, image_size=(width, height)):
                if args.conf_min <= det.confidence <= args.conf_max:
                    found.append({
                        "frame_id": index,
                        "timestamp_s": round(index / fps, 2),
                        "conf": round(det.confidence, 3),
                        "bbox": [int(v) for v in det.bbox],
                        "image": frame.copy(),
                    })
        index += 1
    cap.release()
    detector.close()

    print(f"扫描 {index} 帧，低置信检测 {len(found)} 个")
    tiles = []
    for i, item in enumerate(found[: args.limit]):
        x1, y1, x2, y2 = item["bbox"]
        w, h = x2 - x1, y2 - y1
        px, py = int(w * args.pad), int(h * args.pad)
        box = (max(0, x1 - px), max(0, y1 - py), min(width, x2 + px), min(height, y2 + py))
        crop = item["image"][box[1]:box[3], box[0]:box[2]]
        if crop.size == 0:
            continue
        tile = Image.fromarray(crop[:, :, ::-1]).resize((args.size, args.size))
        draw = ImageDraw.Draw(tile)
        sx, sy = args.size / max(1, crop.shape[1]), args.size / max(1, crop.shape[0])
        draw.rectangle([(x1 - box[0]) * sx, (y1 - box[1]) * sy,
                        (x2 - box[0]) * sx, (y2 - box[1]) * sy],
                       outline=(255, 60, 60), width=2)
        draw.text((3, 3), f"#{i} {item['conf']:.2f}", fill=(255, 255, 0))
        tiles.append(tile)

    if tiles:
        cols = 8
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * args.size, rows * args.size), (20, 20, 20))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols) * args.size, (i // cols) * args.size))
        sheet.save(out / "lowconf.jpg", quality=88)
        print(f"拼图：{out / 'lowconf.jpg'}")

    (out / "lowconf.json").write_text(
        json.dumps([{k: v for k, v in f.items() if k != "image"} for f in found],
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
