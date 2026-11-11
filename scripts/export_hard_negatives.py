"""把误检区域裁成背景图（无标注），作为难负样本加入训练。

    python scripts/export_hard_negatives.py --out data/hard_negatives

为什么裁小图而不是整帧：
整帧里通常还站着真球拍，标成"什么都没有"会把真球拍也教成背景。
只裁误检框周围一小块，保证真球拍不在框里，这样的负样本才是干净的。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _overlaps(a, b) -> float:
    """两个框的相交面积占较小框的比例。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    small = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return inter / small if small > 0 else 0.0


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="导出背景负样本")
    parser.add_argument("--duplicates", default="reports/dup_side/duplicates.json")
    parser.add_argument("--video", default=r"C:\Users\rokmy\Downloads\test_4.mp4")
    parser.add_argument("--out", default="data/hard_negatives")
    parser.add_argument("--pad", type=float, default=0.30, help="误检框四周外扩比例")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=300)
    args = parser.parse_args(argv)

    import cv2

    entries = json.loads(Path(args.duplicates).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.jpg"):
        stale.unlink()

    cap = cv2.VideoCapture(args.video)
    saved = 0
    skipped = 0
    seen_frames: set[int] = set()
    for entry in entries:
        if saved >= args.max_samples:
            break
        frame_id = entry["frame_id"]
        if frame_id in seen_frames:
            continue
        boxes = entry["boxes"]
        confs = entry["confs"]
        if len(boxes) < 2:
            continue
        order = sorted(range(len(confs)), key=lambda i: confs[i])
        fake = boxes[order[0]]        # 置信度低的那个是误检
        real = boxes[order[-1]]       # 置信度高的是真球拍
        # 误检框外扩后不能碰到真球拍，否则会把真球拍教成背景
        fw, fh = fake[2] - fake[0], fake[3] - fake[1]
        pad_x, pad_y = fw * args.pad, fh * args.pad
        crop_box = (fake[0] - pad_x, fake[1] - pad_y, fake[2] + pad_x, fake[3] + pad_y)
        if _overlaps(crop_box, real) > 0.05:
            skipped += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        x1 = max(0, int(crop_box[0]))
        y1 = max(0, int(crop_box[1]))
        x2 = min(frame.shape[1], int(crop_box[2]))
        y2 = min(frame.shape[0], int(crop_box[3]))
        if x2 - x1 < 32 or y2 - y1 < 32:
            continue
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (args.size, args.size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"bg_{saved:04d}.jpg"), crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        seen_frames.add(frame_id)
        saved += 1

    cap.release()
    print(json.dumps({
        "背景图数量": saved,
        "因离真球拍太近而跳过": skipped,
        "输出目录": str(out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
