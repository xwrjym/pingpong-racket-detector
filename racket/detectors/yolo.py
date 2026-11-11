"""ultralytics YOLO 后端（训练见 ``racket.train``）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..tracker import Detection
from .base import DetectorConfig, boxes_to_detections, side_from_class_name


class YoloRacketDetector:
    def __init__(
        self,
        weights: str | Path = "runs/racket_v0.1/weights/best.pt",
        config: DetectorConfig | None = None,
        *,
        names: dict[int, str] | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - 环境缺依赖时的明确提示
            raise RuntimeError(
                "没有找到 ultralytics，请先运行 README 里的安装命令"
            ) from exc
        self.config = config or DetectorConfig()
        self.weights = str(weights)
        if not Path(self.weights).exists():
            raise FileNotFoundError(f"权重不存在：{self.weights}")
        self.model = YOLO(self.weights)
        self.names = names or {int(k): str(v) for k, v in self.model.names.items()}

    # ------------------------------------------------------------------ 推理
    def detect(self, frame: Any, *, image_size: tuple[int, int] | None = None) -> list[Detection]:
        kwargs: dict[str, Any] = {
            "conf": self.config.conf_thr,
            "iou": self.config.iou_thr,
            "imgsz": self.config.imgsz,
            "max_det": self.config.max_det,
            "verbose": False,
        }
        if self.config.device not in ("auto", ""):
            kwargs["device"] = self.config.device
        if self.config.half:
            kwargs["half"] = True
        results = self.model.predict(frame, **kwargs)
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            detections.extend(boxes_to_detections(
                boxes.xyxy.tolist(),
                boxes.conf.tolist(),
                boxes.cls.tolist(),
                self.names,
            ))
        return detections

    # ------------------------------------------------------------------ 部署
    def export_onnx(
        self,
        dest: str | Path,
        *,
        imgsz: int = 640,
        opset: int = 12,
        dynamic: bool = False,
    ) -> str:
        """导出 ONNX 给 RK3588 / onnxruntime 用，返回导出路径。"""
        path = self.model.export(format="onnx", imgsz=imgsz, opset=opset, dynamic=dynamic)
        return str(path)

    def warmup(self, image_size: tuple[int, int] = (1920, 1080)) -> None:
        """跑一次空推理，把首次加载的几百毫秒从延迟统计里排掉。"""
        import numpy as np

        blank = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        self.detect(blank, image_size=image_size)

    def close(self) -> None:
        self.model = None


def describe_classes(names: dict[int, str] | None = None) -> dict[str, Any]:
    names = names or {0: "racket_A", 1: "racket_B", 2: "racket_unknown"}
    return {
        "names": names,
        "side_map": {cid: side_from_class_name(name) for cid, name in names.items()},
    }
