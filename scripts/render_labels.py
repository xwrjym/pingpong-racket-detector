"""把标注或预测画到图上，用于目视检查。

    python scripts/render_labels.py data/videos/test_4/labels_manual.jsonl reports/label_check --frames 0,3,4,7
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = {"A": "#22ff88", "B": "#ffb020", "UNKNOWN": "#a0a0a0"}


def render(labels_path: str, out_dir: str, frames: list[int] | None = None,
           conf_key: str = "side") -> int:
    rows = [json.loads(line) for line in Path(labels_path).open(encoding="utf-8") if line.strip()]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in rows:
        if frames and row["frame_id"] not in frames:
            continue
        if not row.get("boxes"):
            continue
        image = Image.open(row["image_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for box in row["boxes"]:
            x1, y1, x2, y2 = box["bbox_xyxy"]
            side = box.get(conf_key, "UNKNOWN")
            color = COLORS.get(side, "#ff4040")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            label = side
            if box.get("score") is not None:
                label += f" {box['score']:.2f}"
            draw.text((x1 + 5, max(0, y1 - 15)), label, fill=color)
        target = out / f"f{row['frame_id']:06d}.jpg"
        image.save(target, quality=90)
        count += 1
    print(f"渲染 {count} 帧 -> {out}")
    return count


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    labels = argv[0] if argv else "data/videos/test_4/labels_manual.jsonl"
    out_dir = argv[1] if len(argv) > 1 else "reports/label_check"
    frames = None
    if len(argv) > 2 and argv[2].startswith("--frames"):
        value = argv[2].split("=", 1)[-1] if "=" in argv[2] else (argv[3] if len(argv) > 3 else "")
        frames = [int(v) for v in value.split(",") if v.strip()]
    return 0 if render(labels, out_dir, frames) >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
