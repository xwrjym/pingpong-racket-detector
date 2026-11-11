"""把球拍检测结果烧进视频，生成可发给别人看的演示片段。

    python scripts/demo_video.py <weights> --out reports/demo.mp4 --fps 30

抽到 ``--fps`` 帧率逐帧推理（源视频 120 fps，全跑一遍意义不大），
框用跟踪器给的稳定 track_id 上色，左上角显示时间戳与当前帧号。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racket.adapter import AdapterConfig, RacketAdapter      # noqa: E402
from racket.detectors import YoloRacketDetector             # noqa: E402
from racket.detectors.base import DetectorConfig            # noqa: E402

# OpenCV 是 BGR
COLORS = {"A": (136, 255, 34), "B": (32, 176, 255), "UNKNOWN": (176, 176, 176)}


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="生成带检测框的演示视频")
    parser.add_argument("weights")
    parser.add_argument("--video", default=r"C:\Users\rokmy\Downloads\test_4.mp4")
    parser.add_argument("--out", default="reports/demo_v2_full.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--crf", type=int, default=23, help="转码质量，越小越清晰")
    args = parser.parse_args(argv)

    import cv2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频：{args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(1, round(src_fps / args.fps))
    out_fps = src_fps / step
    begin = int(args.start * src_fps)
    finish = total if args.end is None else min(total, int(args.end * src_fps))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_name(out_path.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             out_fps, (width, height))

    detector = YoloRacketDetector(args.weights, DetectorConfig(conf_thr=args.conf,
                                                              imgsz=args.imgsz, device="0"))
    adapter = RacketAdapter(detector, AdapterConfig(session_id="s01", conf_thr=args.conf))

    cap.set(cv2.CAP_PROP_POS_FRAMES, begin)
    index = begin
    written = 0
    hits = 0
    started = time.time()
    while index < finish:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            ts = index / src_fps
            observations = adapter.process_frame(frame, index, ts, (width, height))
            visible = [o for o in observations if o.visible and o.bbox_xyxy]
            hits += len(visible)
            for obs in visible:
                x1, y1, x2, y2 = (int(v) for v in obs.bbox_xyxy)
                color = COLORS.get(obs.side, COLORS["UNKNOWN"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = f"{obs.side} {obs.confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
                cv2.putText(frame, label, (x1 + 4, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (17, 17, 13), 2, cv2.LINE_AA)
            cv2.putText(frame, f"t={ts:6.2f}s  frame={index:6d}", (24, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            written += 1
            if written % 300 == 0:
                speed = written / max(time.time() - started, 1e-6)
                print(f"  {written} 帧已写入（{speed:.1f} fps），当前 t={ts:.1f}s", flush=True)
        index += 1

    cap.release()
    writer.release()
    print(f"写入完成：{written} 帧 -> {raw_path}", flush=True)

    # mp4v 兼容性一般，转成 H.264 方便发微信
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        cmd = [get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(raw_path), "-c:v", "libx264", "-crf", str(args.crf),
               "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out_path)]
        subprocess.run(cmd, check=True)
        raw_path.unlink(missing_ok=True)
        final = out_path
    except Exception as exc:            # 转码失败也保留原始文件
        print(f"转码失败（{exc}），保留 mp4v 版本：{raw_path}")
        final = raw_path

    size_mb = final.stat().st_size / 1048576
    print(f"输出：{final}  {size_mb:.1f} MB  时长 {written / out_fps:.1f}s")
    print(f"平均每帧球拍数 {hits / max(written, 1):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
