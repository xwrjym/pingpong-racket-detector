"""把指定帧号之前的标注标记为"已复核"。

用于给旧标注文件补 reviewed 字段（工具加该字段之前的标注）。
    python scripts/mark_reviewed.py data/videos/test_4/labels_manual.jsonl 60
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    path = Path(argv[0])
    limit = int(argv[1])
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    changed = 0
    for row in rows:
        if row["frame_id"] < limit and not row.get("reviewed"):
            row["reviewed"] = True
            changed += 1
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    reviewed = sum(1 for r in rows if r.get("reviewed"))
    print(f"标记 {changed} 帧 -> 已复核 {reviewed}/{len(rows)} 帧")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
