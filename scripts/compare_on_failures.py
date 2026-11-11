"""在已知误检帧上比较两个模型：同侧重复检测是否减少。

    python scripts/compare_on_failures.py weightsA weightsB

用 ``reports/dup_side/duplicates.json`` 里记录的失败帧做定点对比，
比起重扫整段视频快得多，也直接对得上"有没有修好"。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racket.detectors import YoloRacketDetector     # noqa: E402
from racket.detectors.base import DetectorConfig    # noqa: E402
from racket.geometry import side_from_x             # noqa: E402


def count_duplicates(detector, frames: dict[int, "np.ndarray"], width: int) -> int:
    bad = 0
    for _frame_id, frame in frames.items():
        groups: dict[str, int] = {"A": 0, "B": 0}
        for det in detector.detect(frame, image_size=(width, frame.shape[0])):
            side = side_from_x((det.bbox[0] + det.bbox[2]) / 2, width)
            if side in groups:
                groups[side] += 1
        if groups["A"] > 1 or groups["B"] > 1:
            bad += 1
    return bad


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    import cv2

    conf = float(argv[3]) if len(argv) > 3 else 0.25
    entries = json.loads(Path("reports/dup_side/duplicates.json").read_text(encoding="utf-8"))
    frame_ids = sorted({e["frame_id"] for e in entries})
    video = argv[2] if len(argv) > 2 else r"C:\Users\rokmy\Downloads\test_4.mp4"
    cap = cv2.VideoCapture(video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames: dict[int, object] = {}
    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if ok:
            frames[frame_id] = frame
    cap.release()
    print(f"失败帧 {len(frames)} 个")

    results = {}
    for weights in argv[:1] if argv else []:
        pass
    for weights in [a for a in argv if a.endswith(".pt")]:
        detector = YoloRacketDetector(weights, DetectorConfig(conf_thr=conf, imgsz=640, device="0"))
        bad = count_duplicates(detector, frames, width)
        detector.close()
        results[Path(weights).parent.parent.name or weights] = bad
        print(f"{Path(weights).parent.parent.name} (conf={conf}): 仍然同侧重复 {bad}/{len(frames)} 帧")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
