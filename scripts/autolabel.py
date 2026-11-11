"""用已有模型给未标注的帧预填框，人工只需要改错。

    python scripts/autolabel.py data/videos/test_4/labels_manual.jsonl \
        runs/detect/runs/v1_60frames/weights/best.pt --conf 0.15

只处理 ``reviewed=False`` 的帧，写回时**保持** reviewed=False，所以：
* 训练时仍不会把预填框当真值（build_yolo_dataset 只吃 reviewed=True）；
* 人工在标注工具里保存一次，这一帧才变成已复核。

置信度调低是有意的：漏标比多框更难被发现，多出来的框人一眼就能删。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racket.adapter import AdapterConfig, RacketAdapter      # noqa: E402
from racket.dataset import BoxLabel, FrameLabel, load_frames, save_frames  # noqa: E402
from racket.detectors import YoloRacketDetector             # noqa: E402
from racket.detectors.base import DetectorConfig            # noqa: E402


def main(argv: list[str]) -> int:
    import argparse

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="模型预填未标注帧")
    parser.add_argument("labels", type=Path)
    parser.add_argument("weights", type=str)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--overwrite", action="store_true", help="连已复核帧一起覆盖（危险）")
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image

    frames = load_frames(args.labels)
    targets = [f for f in frames if args.overwrite or not f.reviewed]
    if not targets:
        print("没有需要预填的帧")
        return 0

    detector = YoloRacketDetector(args.weights, DetectorConfig(conf_thr=args.conf,
                                                               imgsz=args.imgsz, device="0"))
    adapter = RacketAdapter(detector, AdapterConfig(session_id="s01", conf_thr=args.conf))

    filled = 0
    boxes_total = 0
    for index, frame in enumerate(frames):
        if not (args.overwrite or not frame.reviewed):
            continue
        image = np.array(Image.open(frame.image_path).convert("RGB"))[:, :, ::-1]
        observations = adapter.process_frame(
            image, frame.frame_id, frame.timestamp_s, frame.image_size
        )
        boxes = [
            BoxLabel(bbox_xyxy=o.bbox_xyxy, side=o.side, visible=True, hard=())
            for o in observations if o.visible and o.bbox_xyxy
        ]
        frames[index] = FrameLabel(
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            timestamp_s=frame.timestamp_s,
            image_path=frame.image_path,
            boxes=boxes,
            image_size=frame.image_size,
            reviewed=False,          # 关键：预填不等于已复核
        )
        filled += 1
        boxes_total += len(boxes)

    save_frames(frames, args.labels)
    print(json.dumps({
        "预填帧数": filled,
        "预填框数": boxes_total,
        "平均每帧": round(boxes_total / max(filled, 1), 2),
        "conf": args.conf,
        "仍需人工复核的帧": sum(1 for f in frames if not f.reviewed),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
