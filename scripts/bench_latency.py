"""测推理延迟：P50 / P95，用来判断 960 的代价能不能接受。

    python scripts/bench_latency.py <weights> --imgsz 640,960 --n 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racket.detectors import YoloRacketDetector     # noqa: E402
from racket.detectors.base import DetectorConfig    # noqa: E402


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="推理延迟")
    parser.add_argument("weights")
    parser.add_argument("--imgsz", default="640,960")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)

    import cv2

    cap = cv2.VideoCapture(r"C:\Users\rokmy\Downloads\test_3.mp4")
    frames = []
    for i in range(args.n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30 + i * 20)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()

    for size in [int(v) for v in args.imgsz.split(",")]:
        detector = YoloRacketDetector(args.weights,
                                      DetectorConfig(conf_thr=0.5, imgsz=size, device=args.device))
        detector.detect(frames[0], image_size=(1920, 1080))     # 预热
        times = []
        for frame in frames:
            start = time.perf_counter()
            detector.detect(frame, image_size=(1920, 1080))
            times.append((time.perf_counter() - start) * 1000)
        detector.close()
        times.sort()
        p50 = times[len(times) // 2]
        p95 = times[int(len(times) * 0.95)]
        print(f"imgsz={size}: P50 {p50:.1f} ms  P95 {p95:.1f} ms  "
              f"（{1000 / p50:.0f} fps 上限）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
