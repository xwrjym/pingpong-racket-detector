"""比较"连续命中确认"开关的效果：假框少了多少、真框有没有被误伤。

    python scripts/probe_confirm.py <weights> --video ... --fps 30 --hits 1,2,3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racket.adapter import AdapterConfig, RacketAdapter      # noqa: E402
from racket.detectors import YoloRacketDetector             # noqa: E402
from racket.detectors.base import DetectorConfig            # noqa: E402


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="连续命中确认的效果")
    parser.add_argument("weights")
    parser.add_argument("--video", default=r"C:\Users\rokmy\Downloads\test_2.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hits", default="1,2,3")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--watch", default="", help="关注的时刻，逗号分隔")
    args = parser.parse_args(argv)

    import cv2

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    step = max(1, round(fps / args.fps))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    watch = {round(float(v), 2) for v in args.watch.split(",") if v.strip()}

    frames = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            frames.append((frame, index, index / fps))
        index += 1
    cap.release()
    print(f"采样 {len(frames)} 帧")

    for hits in [int(v) for v in args.hits.split(",")]:
        detector = YoloRacketDetector(args.weights, DetectorConfig(conf_thr=args.conf,
                                                                   imgsz=640, device="0"))
        adapter = RacketAdapter(detector, AdapterConfig(session_id="s02", conf_thr=args.conf,
                                                        confirm_hits=hits))
        total = 0
        watched: dict[float, int] = {}
        for frame, frame_id, ts in frames:
            obs = [o for o in adapter.process_frame(frame, frame_id, ts, (width, height)) if o.visible]
            total += len(obs)
            for when in watch:
                if abs(ts - when) < 0.02:
                    watched[when] = len(obs)
        adapter.close()
        print(f"confirm_hits={hits}: 共输出 {total} 个观测"
              f"（平均每帧 {total / max(len(frames), 1):.2f}）"
              + ("  关注时刻: " + ", ".join(f"t={k}s→{v}" for k, v in sorted(watched.items()))
                 if watched else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
