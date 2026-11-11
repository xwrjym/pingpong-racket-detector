"""抽帧工具：优先用 opencv，没有就退回 imageio-ffmpeg 自带的 ffmpeg 二进制。

    python -m racket.tools.extract_frames --video D:/racket-data/s01.mp4 \
        --out D:/racket-data/frames/s01 --stride 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def probe_video(path: str | Path) -> dict[str, Any]:
    path = str(path)
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("opencv 打不开该文件")
        info = {
            "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        cap.release()
        info["duration_s"] = round(info["frames"] / info["fps"], 3) if info["fps"] else None
        return info
    except ImportError:
        pass
    # 没有 opencv：只用 ffmpeg 探测，帧率必须手工指定
    return {
        "fps": None,
        "frames": None,
        "width": None,
        "height": None,
        "duration_s": None,
        "note": "没有 opencv，无法自动读帧率；请用 --fps 指定",
    }


def _ffmpeg_exe() -> str:
    from imageio_ffmpeg import get_ffmpeg_exe

    return get_ffmpeg_exe()


def extract_frames(
    video: str | Path,
    out_dir: str | Path,
    *,
    fps: float | None = None,
    stride: int | None = None,
    start_s: float = 0.0,
    end_s: float | None = None,
    max_frames: int | None = None,
    jpeg_quality: int = 92,
) -> dict[str, Any]:
    """抽帧并返回统计。``stride`` 优先于 ``fps``（抽帧间隔更直观）。"""
    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe_video(video)
    source_fps = fps or info.get("fps") or 30.0

    if stride:
        target_fps = source_fps / float(stride)
        step = int(stride)
    else:
        target_fps = float(fps or source_fps)
        step = max(1, int(round(source_fps / target_fps)))

    try:
        import cv2

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"打不开视频：{video}")
        start_frame = int(round(start_s * source_fps))
        end_frame = None if end_s is None else int(round(end_s * source_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_id = start_frame
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if end_frame is not None and frame_id > end_frame:
                break
            if (frame_id - start_frame) % step == 0:
                name = out_dir / f"{frame_id:06d}.jpg"
                cv2.imwrite(str(name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                saved += 1
                if max_frames and saved >= max_frames:
                    break
            frame_id += 1
        cap.release()
        return {
            "engine": "opencv",
            "video": str(video),
            "out_dir": str(out_dir),
            "frames_saved": saved,
            "source_fps": source_fps,
            "target_fps": round(target_fps, 4),
            "step": step,
            "resolution": [info.get("width"), info.get("height")],
        }
    except ImportError:
        pass

    # 退回 ffmpeg：只保留按 fps 抽帧的路径
    cmd = [_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"]
    if start_s:
        cmd += ["-ss", str(start_s)]
    cmd += ["-i", str(video)]
    if end_s:
        cmd += ["-to", str(end_s - start_s)]
    vf = f"fps={target_fps:.6f}"
    cmd += ["-vf", vf, "-qscale:v", "3", str(out_dir / "%06d.jpg")]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 抽帧失败：{result.stderr[-800:]}")
    saved = len(list(out_dir.glob("*.jpg")))
    return {
        "engine": "ffmpeg",
        "video": str(video),
        "out_dir": str(out_dir),
        "frames_saved": saved,
        "source_fps": source_fps,
        "target_fps": round(target_fps, 4),
        "step": step,
        "resolution": [info.get("width"), info.get("height")],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="视频抽帧")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--stride", type=int, default=None, help="每 N 帧取一帧")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args(argv)

    if args.probe_only:
        print(json.dumps(probe_video(args.video), ensure_ascii=False, indent=2))
        return 0
    info = extract_frames(
        args.video, args.out, fps=args.fps, stride=args.stride,
        start_s=args.start, end_s=args.end, max_frames=args.max_frames,
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
