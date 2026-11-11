"""对标注文件里的**全部已复核帧**做产品链路评估（不切分、不抽样）。

    python scripts/eval_all.py <weights> data/videos/test_2/labels_s02.jsonl

和 ``eval_real.py`` 的区别：后者只评时间切分出来的那一段，这个评整份文件，
用于"整场测试集"的口径。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racket.adapter import AdapterConfig, RacketAdapter      # noqa: E402
from racket.dataset import load_frames                       # noqa: E402
from racket.detectors import YoloRacketDetector              # noqa: E402
from racket.detectors.base import DetectorConfig             # noqa: E402
from racket.eval import PredBox, PredFrame, evaluate_detection  # noqa: E402


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    weights = argv[0]
    labels = argv[1]
    conf = float(argv[2]) if len(argv) > 2 else 0.5
    max_per_side = int(argv[3]) if len(argv) > 3 else 1
    imgsz = int(argv[4]) if len(argv) > 4 else 640

    import numpy as np
    from PIL import Image

    frames = [f for f in load_frames(labels) if f.reviewed]
    detector = YoloRacketDetector(weights, DetectorConfig(conf_thr=conf, imgsz=imgsz, device="0"))
    adapter = RacketAdapter(detector, AdapterConfig(session_id=frames[0].session_id,
                                                   conf_thr=conf, max_per_side=max_per_side))

    predicted: list[PredFrame] = []
    for frame in frames:
        image = np.array(Image.open(frame.image_path).convert("RGB"))[:, :, ::-1]
        observations = adapter.process_frame(image, frame.frame_id, frame.timestamp_s,
                                             frame.image_size)
        predicted.append(PredFrame(
            frame.session_id, frame.frame_id, frame.timestamp_s,
            [PredBox(o.bbox_xyxy, o.confidence or 0.0, o.side)
             for o in observations if o.visible],
        ))
    adapter.close()

    result = evaluate_detection(frames, predicted)
    summary = {
        "frames": len(frames),
        "conf": conf,
        "mAP50": result["mAP50"],
        "frame_recall": result["frame_recall"],
        "side_accuracy": result["side_accuracy"],
        "side_unknown_rate": result["side_unknown_rate"],
        "precision_recall": result["precision_recall"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
