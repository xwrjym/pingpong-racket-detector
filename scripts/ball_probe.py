"""粗检测白球：用于定位"击球时刻"，不是正式的球检测模块。

    python scripts/ball_probe.py --video ... --fps 60 --limit 300

做法：高亮低饱和 + 帧间运动门控 + 连通域面积/长宽比筛选，只在球台区域内找。
输出检出率，用来判断这条路能不能支撑击球时刻分析。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def find_ball(frame, prev_gray, roi):
    import cv2
    import numpy as np

    x1, y1, x2, y2 = roi
    patch = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 180) & (hsv[:, :, 1] < 80)).astype(np.uint8)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        motion = (cv2.absdiff(gray, prev_gray) > 18).astype(np.uint8)
        mask &= motion
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if not (8 <= area <= 400):
            continue
        if w > 4 * max(h, 1) or h > 4 * max(w, 1):
            continue
        cx, cy = centroids[i]
        score = area
        if best is None or score > best[0]:
            best = (score, cx + x1, cy + y1)
    return None if best is None else (best[1], best[2], gray)


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=r"C:\Users\rokmy\Downloads\test_1.mp4")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args(argv)

    import cv2

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    step = max(1, round(src_fps / args.fps))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 球台大致区域：画面中部往下一带
    roi = (int(width * 0.05), int(height * 0.40), int(width * 0.98), int(height * 0.95))

    prev_gray = None
    seen = 0
    hits = 0
    index = 0
    while index < args.limit * step:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            found = find_ball(frame, prev_gray, roi)
            if found:
                hits += 1
            seen += 1
        index += 1
    cap.release()
    print(f"采样 {seen} 帧，检出球 {hits} 帧，检出率 {hits / max(seen, 1):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
