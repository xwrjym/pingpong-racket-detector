"""统计整段视频里每一侧的球拍检出覆盖，找成段的漏检区间。

    python scripts/coverage_report.py <weights> --video ...

没有击球标注时，这是判断"击球时机有没有漏拍"最实在的办法：
漏检如果只是零散单帧，影响有限；如果能连成 0.5 秒以上的段，
那段里发生的击球就一定没有球拍证据。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racket.adapter import AdapterConfig, RacketAdapter      # noqa: E402
from racket.detectors import YoloRacketDetector             # noqa: E402
from racket.detectors.base import DetectorConfig            # noqa: E402


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="检出覆盖率与漏检区间")
    parser.add_argument("weights")
    parser.add_argument("--video", default=r"C:\Users\rokmy\Downloads\test_1.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--out", default="reports/coverage_test1")
    args = parser.parse_args(argv)

    import cv2
    import numpy as np

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    step = max(1, round(src_fps / args.fps))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_out = src_fps / step

    detector = YoloRacketDetector(args.weights, DetectorConfig(conf_thr=args.conf,
                                                               imgsz=args.imgsz, device="0"))
    adapter = RacketAdapter(detector, AdapterConfig(session_id="s01", conf_thr=args.conf))

    times: list[float] = []
    seen: dict[str, list[bool]] = {"A": [], "B": []}
    activity: list[float] = []
    prev_gray = None
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            ts = index / src_fps
            observations = adapter.process_frame(frame, index, ts, (width, height))
            present = {o.side for o in observations if o.visible}
            times.append(ts)
            for side in ("A", "B"):
                seen[side].append(side in present)
            # 画面活跃度：比赛进行中两人都在动，死球间歇几乎静止，
            # 用它把"没在打球"的帧排除掉，否则覆盖率被间歇稀释。
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                activity.append(float(np.abs(gray.astype("int16")
                                             - prev_gray.astype("int16")).mean()))
            prev_gray = gray
        index += 1
    cap.release()
    adapter.close()

    stats: dict[str, object] = {"frames": len(times), "fps": round(fps_out, 2),
                                "conf": args.conf, "imgsz": args.imgsz}
    for side in ("A", "B"):
        flags = seen[side]
        rate = sum(flags) / max(len(flags), 1)
        # 最长连续缺失段
        best_len = cur = 0
        best_start = cur_start = 0.0
        gaps = []
        for i, flag in enumerate(flags):
            if flag:
                if cur:
                    gaps.append((cur_start, times[i - 1], cur))
                cur = 0
            else:
                if cur == 0:
                    cur_start = times[i]
                cur += 1
        if cur:
            gaps.append((cur_start, times[-1], cur))
        gaps.sort(key=lambda g: -g[2])
        long_gaps = [(round(a, 2), round(b, 2), round(b - a, 2)) for a, b, n in gaps[:10]]
        stats[side] = {
            "检出率": round(rate, 4),
            "最长漏检_s": round(max((b - a for a, b, _ in gaps), default=0.0), 2),
            "超过0.5s的漏检段数": sum(1 for a, b, _ in gaps if b - a >= 0.5),
            "前10个最长漏检段": long_gaps,
        }

    both = sum(1 for a, b in zip(seen["A"], seen["B"]) if a and b) / max(len(times), 1)
    stats["两侧同时检出率"] = round(both, 4)

    # 只统计"在打球"的帧：画面活跃度前 55%
    if activity:
        arr = np.array(activity, dtype=float)
        threshold = float(np.percentile(arr, 45))
        active_idx = [i + 1 for i, value in enumerate(activity) if value >= threshold]
        active_stats: dict[str, object] = {
            "活跃阈值": round(threshold, 2),
            "活跃帧数": len(active_idx),
            "占比": round(len(active_idx) / max(len(times), 1), 3),
        }
        for side in ("A", "B"):
            flags = [seen[side][i] for i in active_idx]
            active_stats[side] = round(sum(flags) / max(len(flags), 1), 4)
        active_stats["两侧同时"] = round(
            sum(1 for i in active_idx if seen["A"][i] and seen["B"][i]) / max(len(active_idx), 1), 4)
        stats["活跃帧检出率"] = active_stats
    (out_dir / "coverage.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    # 时间轴图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 2.6), dpi=120)
        for row, side in enumerate(("A", "B")):
            flags = np.array(seen[side], dtype=float)
            ax.fill_between(times, row + 0.1, row + 0.9, where=flags > 0,
                            step="post", color="#22c55e", linewidth=0)
            ax.fill_between(times, row + 0.1, row + 0.9, where=flags <= 0,
                            step="post", color="#ef4444", linewidth=0)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["A 左半台", "B 右半台"])
        ax.set_xlabel("时间 (s)")
        ax.set_title("球拍检出覆盖（绿=检出，红=漏检）")
        ax.set_xlim(times[0], times[-1])
        fig.tight_layout()
        fig.savefig(out_dir / "coverage_timeline.png")
        print(f"时间轴图：{out_dir / 'coverage_timeline.png'}")
    except Exception as exc:
        print(f"画图失败（不影响数据）：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
