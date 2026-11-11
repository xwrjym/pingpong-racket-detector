"""onnxruntime 后端：给板端和"不想装 ultralytics"的机器用。

只依赖 numpy + onnxruntime + Pillow，输出与 YOLO 后端完全一致的
``Detection`` 列表，所以适配器不需要知道跑的是哪个后端。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..tracker import Detection
from .base import DetectorConfig, side_from_class_name


def _letterbox(image: np.ndarray, size: int, pad_value: int = 114) -> tuple[np.ndarray, float, int, int]:
    """等比缩放并居中填充，返回 (blob, ratio, pad_x, pad_y)。"""
    from PIL import Image

    height, width = image.shape[:2]
    ratio = min(size / width, size / height)
    new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
    pil = Image.fromarray(image[:, :, ::-1])  # BGR -> RGB
    pil = pil.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (pad_value, pad_value, pad_value))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(pil, (pad_x, pad_y))
    blob = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return blob[None, ...], ratio, pad_x, pad_y


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.clip(area_i + area_r - inter, 1e-9, None)
        order = rest[iou <= iou_thr]
    return keep


class OnnxRacketDetector:
    def __init__(
        self,
        model_path: str | Path,
        config: DetectorConfig | None = None,
        *,
        names: dict[int, str] | None = None,
        providers: Sequence[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("没有找到 onnxruntime：pip install onnxruntime") from exc
        self.config = config or DetectorConfig()
        self.model_path = str(model_path)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"ONNX 模型不存在：{self.model_path}")
        available = ort.get_available_providers()
        wanted = list(providers) if providers else [
            p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available
        ]
        self.session = ort.InferenceSession(self.model_path, providers=wanted or available)
        self.input_name = self.session.get_inputs()[0].name
        self.names = names or {0: "racket_A", 1: "racket_B", 2: "racket_unknown"}

    def detect(self, frame: Any, *, image_size: tuple[int, int] | None = None) -> list[Detection]:
        if not isinstance(frame, np.ndarray):
            raise TypeError("ONNX 后端需要 BGR numpy 图像")
        height, width = frame.shape[:2]
        blob, ratio, pad_x, pad_y = _letterbox(frame, self.config.imgsz)
        outputs = self.session.run(None, {self.input_name: blob})
        predictions = np.asarray(outputs[0])
        if predictions.ndim == 3:
            predictions = predictions[0]
        # YOLOv8 导出格式：(4 + nc, N) -> (N, 4 + nc)
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        if predictions.shape[1] <= 4:
            return []

        boxes_cxcywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)

        mask = scores >= self.config.conf_thr
        if not np.any(mask):
            return []
        boxes_cxcywh = boxes_cxcywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # 中心点格式 -> 角点格式，并还原到原图坐标
        cx, cy, bw, bh = boxes_cxcywh.T
        x1 = (cx - bw / 2 - pad_x) / ratio
        y1 = (cy - bh / 2 - pad_y) / ratio
        x2 = (cx + bw / 2 - pad_x) / ratio
        y2 = (cy + bh / 2 - pad_y) / ratio
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height - 1)

        detections: list[Detection] = []
        for cls in np.unique(class_ids):
            idx = np.where(class_ids == cls)[0]
            for keep in _nms(boxes[idx], scores[idx], self.config.iou_thr)[: self.config.max_det]:
                i = int(idx[keep])
                name = self.names.get(int(cls), str(int(cls)))
                detections.append(Detection(
                    bbox=tuple(float(v) for v in boxes[i]),
                    confidence=float(scores[i]),
                    side=side_from_class_name(name),
                    class_id=int(cls),
                    class_name=name,
                ))
        detections.sort(key=lambda d: -d.confidence)
        return detections

    def close(self) -> None:
        self.session = None
