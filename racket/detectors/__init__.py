"""检测后端：默认 ultralytics YOLO（路线 A 独立球拍模型）。"""

from .base import Detector, DetectorConfig
from .fake import StaticDetector

__all__ = ["Detector", "DetectorConfig", "StaticDetector", "YoloRacketDetector"]


def __getattr__(name: str):  # 懒加载，避免没装 ultralytics 时整个包 import 失败
    if name == "YoloRacketDetector":
        from .yolo import YoloRacketDetector

        return YoloRacketDetector
    raise AttributeError(name)
