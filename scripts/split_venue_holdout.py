"""按"场次 + 帧区间"切分：把某个场次的前段并入训练、后段留作测试。

    python scripts/split_venue_holdout.py <labels.jsonl> <test_session> <holdout_ratio> <dest>

用途：在只有两处场地、又想把新场地纳入训练时，至少还能在新场地的**时间后段**上
留一个检验点。注意这是同场地时间切分，指标比"整场留出"乐观。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from racket.dataset import export_yolo, load_frames           # noqa: E402


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("labels")
    parser.add_argument("test_session", help="要切出留出段的场次，如 s06")
    parser.add_argument("holdout_ratio", type=float, help="该场次后段留出的比例，如 0.3")
    parser.add_argument("dest")
    parser.add_argument("--background-dir", default="data/backgrounds_s01s02")
    args = parser.parse_args(argv)

    frames = [f for f in load_frames(args.labels) if f.reviewed]
    test_session_frames = sorted([f for f in frames if f.session_id == args.test_session],
                                key=lambda f: f.frame_id)
    if not test_session_frames:
        raise SystemExit(f"没有 {args.test_session} 的帧")

    cut = int(len(test_session_frames) * (1 - args.holdout_ratio))
    head = test_session_frames[:cut]
    tail = test_session_frames[cut:]

    train_pool = [f for f in frames if f.session_id != args.test_session] + head
    train_pool.sort(key=lambda f: (f.session_id, f.frame_id))
    n_val = max(1, round(len(train_pool) * 0.12))
    val = train_pool[:n_val]
    train = train_pool[n_val:]

    info = export_yolo(train + val + tail, args.dest,
                       splits={"train": train, "val": val, "test": tail},
                       backgrounds=sorted(Path(args.background_dir).glob("*.jpg"))
                       if args.background_dir else ())
    summary = {
        "train": len(train), "val": len(val), "test": len(tail),
        "留出场次": args.test_session,
        "留出帧区间": [tail[0].frame_id, tail[-1].frame_id],
        "并入训练的新场次帧数": len(head),
        "note": "同场地时间切分，指标偏乐观",
        "data_yaml": info["data_yaml"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
