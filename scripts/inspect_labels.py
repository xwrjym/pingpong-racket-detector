"""看一眼标注质量：帧数、框数、框尺寸、侧别标签和位置是否一致。

    python scripts/inspect_labels.py data/videos/test_4/labels_manual.jsonl

校验项里最容易出错的是「侧别标错」和「框太小（只框了胶皮）」，
这两项直接影响训练出来的模型能不能用。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    path = Path(argv[0] if argv else "data/videos/test_4/labels_manual.jsonl")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    labeled = [r for r in rows if r.get("boxes")]
    print(f"文件：{path}")
    print(f"已标注 {len(labeled)} / {len(rows)} 帧\n")

    total = 0
    side_conflict = 0
    tiny = 0
    for row in labeled:
        width = row.get("image_size", [1920, 1080])[0]
        mid = width / 2
        print(f"帧 {row['frame_id']:>3}  t={row['timestamp_s']:>6.1f}s  框数={len(row['boxes'])}")
        for box in row["boxes"]:
            x1, y1, x2, y2 = box["bbox_xyxy"]
            w, h = x2 - x1, y2 - y1
            cx = (x1 + x2) / 2
            by_pos = "A" if cx < mid else "B"
            notes = []
            if box["side"] in ("A", "B") and box["side"] != by_pos:
                notes.append(f"侧别标了{box['side']}但位置在{by_pos}")
                side_conflict += 1
            if w < 40 or h < 40:
                notes.append("框偏小")
                tiny += 1
            total += 1
            print(f"    {box['side']}  {w:>4.0f}×{h:>4.0f}px  中心({cx:>5.0f},{(y1 + y2) / 2:>4.0f})"
                  f"  hard={box.get('hard', [])}" + (f"   ⚠ {'；'.join(notes)}" if notes else ""))

    print(f"\n合计 {total} 个框；侧别与位置冲突 {side_conflict} 个；小于 40px 的框 {tiny} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
