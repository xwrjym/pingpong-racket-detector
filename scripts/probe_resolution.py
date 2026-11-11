"""比较不同输入分辨率对同一帧的影响。

    python scripts/probe_resolution.py <weights> --video ... --times 18,4 --imgsz 640,960,1280

手和球拍在 640 下都只有二十几像素，放大后纹理和形状差异才显现。
这个脚本用来判断"空手被认成球拍"是不是分辨率不够导致的。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racket.detectors import YoloRacketDetector     # noqa: E402
from racket.detectors.base import DetectorConfig    # noqa: E402
from racket.geometry import side_from_x             # noqa: E402


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="分辨率对比")
    parser.add_argument("weights")
    parser.add_argument("--video", default=r"C:\Users\rokmy\Downloads\test_2.mp4")
    parser.add_argument("--times", default="4,18")
    parser.add_argument("--imgsz", default="640,960,1280")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args(argv)

    import cv2

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames = {}
    for value in args.times.split(","):
        frame_id = int(float(value) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if ok:
            frames[float(value)] = frame
    cap.release()

    for size in [int(v) for v in args.imgsz.split(",")]:
        detector = YoloRacketDetector(args.weights,
                                      DetectorConfig(conf_thr=args.conf, imgsz=size, device="0"))
        print(f"\nimgsz={size}")
        for when, frame in sorted(frames.items()):
            detections = detector.detect(frame, image_size=(width, frame.shape[0]))
            items = []
            for det in detections:
                cx = (det.bbox[0] + det.bbox[2]) / 2
                items.append(f"{side_from_x(cx, width)}@{det.confidence:.2f}"
                             f"({int(det.bbox[2] - det.bbox[0])}px)")
            print(f"  t={when:>4.0f}s: {len(detections)} 个 -> {' '.join(items) or '无'}")
        detector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
