"""按物理半台重算每个框的侧别（换边后同一支球拍会换 side，这是设计意图）。

    python scripts/fix_sides.py data/videos/test_4/labels_manual.jsonl

规则与 adapter 完全一致：中心 x 落在网线附近死区内判 UNKNOWN，否则左半台 A / 右半台 B。
--dry-run 只看不改。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racket.geometry import side_from_x   # noqa: E402

NET_X_RATIO = 0.5
DEAD_ZONE_RATIO = 0.02


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = [a for a in argv if not a.startswith("--")]
    dry_run = "--dry-run" in argv
    path = Path(args[0] if args else "data/videos/test_4/labels_manual.jsonl")

    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    changed = 0
    details = []
    for row in rows:
        width = row.get("image_size", [1920, 1080])[0]
        mid = NET_X_RATIO * width
        for box in row.get("boxes", []):
            x1, _, x2, _ = box["bbox_xyxy"]
            cx = (x1 + x2) / 2
            want = side_from_x(cx, width, net_x=mid, dead_zone_px=DEAD_ZONE_RATIO * width)
            if box["side"] != want:
                details.append((row["frame_id"], box["side"], want, round(cx)))
                box["side"] = want
                changed += 1

    if not dry_run and changed:
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    for frame_id, old, new, cx in details[:20]:
        print(f"  帧 {frame_id:>3}  中心x={cx:>5}  {old} -> {new}")
    if len(details) > 20:
        print(f"  ... 另有 {len(details) - 20} 处")
    print(f"{'检查' if dry_run else '修正'}完成：{changed} 个框的侧别按位置重算")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
