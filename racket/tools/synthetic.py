"""合成测试画面：只用于验证链路能跑通，**不能用于报告任何精度指标**。

真实指标必须在真机视频上算（方案 §12.2）。这里画几张有球台、有球拍、有运动
模糊的图，用来检查抽帧 -> 预标注 -> 导出 -> 评估 这条流水线不报错、
框的位置和侧别也对得上。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFilter


def draw_frame(
    width: int,
    height: int,
    rackets: Sequence[dict[str, Any]],
    *,
    blur: bool = False,
    noise: float = 0.0,
    seed: int = 0,
) -> Image.Image:
    rng = random.Random(seed)
    image = Image.new("RGB", (width, height), (18, 24, 34))
    draw = ImageDraw.Draw(image)

    # 球台：深绿 + 白色边线 + 中线（网）
    table = [width * 0.08, height * 0.45, width * 0.92, height * 0.95]
    draw.rectangle(table, fill=(16, 78, 58), outline=(210, 210, 210), width=3)
    draw.line([(width / 2, table[1]), (width / 2, table[3])], fill=(225, 225, 225), width=4)

    for racket in rackets:
        cx, cy = racket["center"]
        radius = racket["radius"]
        handle_deg = racket.get("handle_deg", 90.0)
        rubber = racket.get("color", (200, 40, 40))
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                     fill=rubber, outline=(30, 20, 20), width=3)
        # 拍柄
        hx = cx + math.cos(math.radians(handle_deg)) * radius * 1.5
        hy = cy + math.sin(math.radians(handle_deg)) * radius * 1.5
        draw.line([(cx, cy), (hx, hy)], fill=(150, 110, 70), width=int(max(6, radius // 4)))

    if blur:
        image = image.filter(ImageFilter.GaussianBlur(1.1))
    if noise:
        pixels = image.load()
        for _ in range(int(width * height * noise)):
            x, y = rng.randrange(width), rng.randrange(height)
            pixels[x, y] = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
    return image


def generate(out_dir: str | Path, *, sessions: int = 3, frames_per_session: int = 6,
             width: int = 1280, height: int = 720, seed: int = 0) -> dict[str, Any]:
    """生成 frames/ gt_labels.jsonl manifest.jsonl 三样东西。"""
    out_dir = Path(out_dir)
    frames_root = out_dir / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    gt_lines: list[str] = []
    manifest_lines: list[str] = []
    for s in range(sessions):
        session_id = f"s{s + 1:02d}"
        (frames_root / session_id).mkdir(parents=True, exist_ok=True)
        for i in range(frames_per_session):
            frame_id = i * 15
            t = frame_id / 30.0
            # 一只球拍从左半台扫到右半台附近，制造真实运动
            phase = i / max(1, frames_per_session - 1)
            cx = width * (0.22 + 0.16 * math.sin(phase * math.pi * 2 + s))
            cy = height * (0.58 + 0.10 * math.cos(phase * math.pi * 3))
            radius = width * (0.030 + 0.004 * s)
            rackets = [{
                "center": (cx, cy), "radius": radius,
                "handle_deg": 90.0 + 40 * math.sin(phase * 6),
                "color": (200, 40, 40) if s % 2 == 0 else (40, 60, 200),
            }]
            image = draw_frame(width, height, rackets,
                               blur=phase > 0.5, noise=0.0005, seed=seed + frame_id)
            path = frames_root / session_id / f"{frame_id:06d}.jpg"
            image.save(path, quality=88)

            side = "A" if cx < width / 2 else "B"
            pad = radius * 1.7
            box = [max(0.0, cx - pad), max(0.0, cy - pad),
                   min(float(width), cx + pad), min(float(height), cy + pad)]
            gt_lines.append(json.dumps({
                "session_id": session_id,
                "frame_id": frame_id,
                "timestamp_s": round(t, 4),
                "image_path": str(path.resolve()),
                "image_size": [width, height],
                "boxes": [{"bbox_xyxy": [round(v, 1) for v in box],
                           "side": side, "visible": True, "hard": []}],
            }, ensure_ascii=False))
            manifest_lines.append(json.dumps({
                "session_id": session_id,
                "frame_id": frame_id,
                "timestamp_s": round(t, 4),
                "image_path": str(path.resolve()),
                "image_width": width,
                "image_height": height,
            }, ensure_ascii=False))

    (out_dir / "gt_labels.jsonl").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    (out_dir / "manifest.jsonl").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return {
        "out_dir": str(out_dir),
        "sessions": sessions,
        "frames": sessions * frames_per_session,
        "gt_labels": str(out_dir / "gt_labels.jsonl"),
        "manifest": str(out_dir / "manifest.jsonl"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成合成测试画面")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--frames-per-session", type=int, default=6)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args(argv)
    info = generate(args.out, sessions=args.sessions,
                    frames_per_session=args.frames_per_session,
                    width=args.width, height=args.height)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
