"""把漏检按目标大小 / 侧别 / 位置拆开，找出主要缺口在哪一类。

    python scripts/recall_breakdown.py <weights> data/videos/test_3/labels_s03.jsonl

只有知道漏的是"小目标"还是"被遮挡"，才知道该加分辨率、加数据还是换更大模型。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racket.adapter import AdapterConfig, RacketAdapter      # noqa: E402
from racket.dataset import load_frames                       # noqa: E402
from racket.detectors import YoloRacketDetector              # noqa: E402
from racket.detectors.base import DetectorConfig             # noqa: E402
from racket.geometry import iou                              # noqa: E402


def bucket_by_size(width: float) -> str:
    if width < 50:
        return "小 <50px"
    if width < 90:
        return "中 50-90px"
    return "大 ≥90px"


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("weights")
    parser.add_argument("labels")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--out", default="reports/recall_breakdown.json")
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image

    frames = [f for f in load_frames(args.labels) if f.reviewed]
    detector = YoloRacketDetector(args.weights,
                                  DetectorConfig(conf_thr=args.conf, imgsz=args.imgsz, device="0"))
    adapter = RacketAdapter(detector, AdapterConfig(session_id="s03", conf_thr=args.conf))

    stats: dict[str, dict[str, int]] = {}

    def bump(group: str, key: str, hit: bool, detail: str = "") -> None:
        for g in (group, "全部"):
            stats.setdefault(g, {})
            stats[g].setdefault(key, 0)
            stats[g][key] += 1
            if hit:
                stats[g].setdefault(key + "_命中", 0)
                stats[g][key + "_命中"] += 1

    for frame in frames:
        image = np.array(Image.open(frame.image_path).convert("RGB"))[:, :, ::-1]
        observations = adapter.process_frame(image, frame.frame_id, frame.timestamp_s,
                                             frame.image_size)
        preds = [o for o in observations if o.visible and o.bbox_xyxy]
        used: set[int] = set()
        height = frame.image_size[1]
        for gt in frame.visible_boxes:
            x1, y1, x2, y2 = gt.bbox_xyxy
            best, best_iou = -1, 0.0
            for i, obs in enumerate(preds):
                if i in used:
                    continue
                value = iou(gt.bbox_xyxy, obs.bbox_xyxy)
                if value > best_iou:
                    best, best_iou = i, value
            hit = best >= 0 and best_iou >= 0.5
            if hit:
                used.add(best)
            bump("按大小", bucket_by_size(x2 - x1), hit)
            bump("按侧别", gt.side, hit)
            cy = (y1 + y2) / 2 / height
            band = "上 1/3" if cy < 1 / 3 else ("中 1/3" if cy < 2 / 3 else "下 1/3")
            bump("按高度", band, hit)
    detector.close()

    summary = {}
    for group, values in stats.items():
        rows = {}
        for key in sorted(k for k in values if not k.endswith("_命中")):
            total = values[key]
            hits = values.get(key + "_命中", 0)
            rows[key] = {"总数": total, "命中": hits,
                         "召回": round(hits / total, 3) if total else 0.0}
        summary[group] = rows
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
