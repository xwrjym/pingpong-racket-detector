"""对指定帧跑推理并可视化，用于目视验收。

    python scripts/predict_frames.py <weights> 5,60,125,140 --out reports/check
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racket.adapter import AdapterConfig, RacketAdapter        # noqa: E402
from racket.dataset import load_frames                        # noqa: E402
from racket.detectors import YoloRacketDetector               # noqa: E402
from racket.detectors.base import DetectorConfig              # noqa: E402

COLORS = {"A": (34, 255, 136), "B": (255, 176, 32), "UNKNOWN": (170, 170, 170)}


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    weights = argv[0]
    wanted = [int(v) for v in argv[1].split(",")] if len(argv) > 1 else [0, 60, 120, 140]
    out = Path(argv[2] if len(argv) > 2 else "reports/check")
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from PIL import Image, ImageDraw

    frames = {f.frame_id: f for f in load_frames("data/videos/test_4/labels_manual.jsonl")}
    detector = YoloRacketDetector(weights, DetectorConfig(conf_thr=0.25, imgsz=640, device="0"))
    adapter = RacketAdapter(detector, AdapterConfig(session_id="s01", conf_thr=0.25))

    for frame_id in wanted:
        frame = frames.get(frame_id)
        if frame is None:
            continue
        image = Image.open(frame.image_path).convert("RGB")
        array = np.array(image)[:, :, ::-1]
        observations = adapter.process_frame(array, frame.frame_id, frame.timestamp_s,
                                             frame.image_size)
        draw = ImageDraw.Draw(image)
        for box in frame.visible_boxes:      # GT 用细白线，方便对照
            draw.rectangle(box.bbox_xyxy, outline=(255, 255, 255), width=2)
        for obs in observations:
            if not obs.visible or not obs.bbox_xyxy:
                continue
            color = COLORS.get(obs.side, COLORS["UNKNOWN"])
            draw.rectangle(obs.bbox_xyxy, outline=color, width=5)
            draw.text((obs.bbox_xyxy[0] + 6, max(0, obs.bbox_xyxy[1] - 18)),
                      f"{obs.side} {obs.confidence:.2f}", fill=color)
        image.save(out / f"pred_{frame_id:06d}.jpg", quality=88)
        print(f"帧 {frame_id}: GT {len(frame.visible_boxes)} 框, 预测 "
              f"{sum(1 for o in observations if o.visible)} 框")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
