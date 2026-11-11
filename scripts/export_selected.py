"""按编号导出选定的低置信检测裁剪，作为背景负样本。

    python scripts/export_selected.py reports/lowconf_t2/lowconf.json 0,1,3,9,10 ...

编号来自 ``mine_lowconf.py`` 生成的拼图，由人眼确认"这块里没有球拍"。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(argv) < 2:
        print("用法：export_selected.py <lowconf.json> <0,1,3,...> [输出目录] [视频]")
        return 2
    import cv2

    data_path = Path(argv[0])
    picked = [int(v) for v in argv[1].split(",") if v.strip()]
    out = Path(argv[2] if len(argv) > 2 else "data/hand_negatives")
    video = argv[3] if len(argv) > 3 else r"C:\Users\rokmy\Downloads\test_2.mp4"
    pad = 0.6

    entries = json.loads(data_path.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.jpg"):
        stale.unlink()

    cap = cv2.VideoCapture(video)
    saved = 0
    for i in picked:
        if i >= len(entries):
            continue
        item = entries[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, item["frame_id"])
        ok, frame = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = item["bbox"]
        w, h = x2 - x1, y2 - y1
        px, py = int(w * pad), int(h * pad)
        bx1, by1 = max(0, x1 - px), max(0, y1 - py)
        bx2 = min(frame.shape[1], x2 + px)
        by2 = min(frame.shape[0], y2 + py)
        crop = frame[by1:by2, bx1:bx2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"hand_{saved:03d}.jpg"), crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        saved += 1
    cap.release()
    print(f"导出 {saved} 张背景图 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
