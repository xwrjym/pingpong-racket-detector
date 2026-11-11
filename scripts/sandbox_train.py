"""沙箱内跑训练的入口：绕开 ultralytics 的多进程标签缓存。

ultralytics 首次扫描数据集时用 ``ThreadPool``（底层走 multiprocessing 的
命名管道），DSH 沙箱禁止创建管道，会报 ``PermissionError: [WinError 5]``。
这里把线程池换成一个顺序执行的替身，并强制 ``workers=0``。

**只用于沙箱自检**；在普通机器上直接用 ``python -m racket.train`` 即可。
"""

from __future__ import annotations

import sys


class _SequentialPool:
    """ThreadPool 的顺序替身，接口只实现数据集扫描用到的那几个。"""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self) -> "_SequentialPool":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def imap_unordered(self, func, iterable, chunksize: int = 1):
        return iter([func(item) for item in iterable])

    def imap(self, func=None, iterable=(), *args, **kwargs):
        return iter([func(item) for item in iterable])

    def map(self, func, iterable, chunksize: int = 1):
        return [func(item) for item in iterable]


def patch() -> None:
    import ultralytics.data.dataset as ds

    ds.ThreadPool = _SequentialPool  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> int:
    import json
    from pathlib import Path

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    patch()

    from racket.train import build_yolo_dataset, evaluate, train

    args = list(argv if argv is not None else sys.argv[1:])
    labels = args[0] if args else "data/selfcheck/synthetic/gt_labels.jsonl"
    epochs = int(args[1]) if len(args) > 1 else 3
    model = args[2] if len(args) > 2 else "yolov8n.yaml"
    name = args[3] if len(args) > 3 else "smoke"
    imgsz = int(args[4]) if len(args) > 4 else 320
    batch = int(args[5]) if len(args) > 5 else 4
    patience = int(args[6]) if len(args) > 6 else 20

    info = build_yolo_dataset(labels, "data/yolo_smoke", seed=1,
                              split_mode=args[7] if len(args) > 7 else "time",
                              background_dir=(args[8] if len(args) > 8 else None))
    print("splits:", info["split_frames"], "leakage:", info["leakage"],
          "unreviewed_skipped:", info["unreviewed_skipped"],
          "backgrounds:", info.get("backgrounds"))
    result = train(
        info["data_yaml"],
        model=model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        name=name,
        project="runs",
        patience=patience,
        extra={"plots": False, "verbose": False, "workers": 0, "val": False},
    )
    print("TRAIN:", json.dumps(result, ensure_ascii=False))
    if result["exists"]:
        for split in ("test", "train"):
            if not info["split_frames"].get(split):
                continue
            metrics = evaluate(result["best_weights"], info["data_yaml"],
                               split=split, imgsz=imgsz, device=0)
            print(f"EVAL[{split}]:", json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
