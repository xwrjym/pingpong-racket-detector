"""球拍检测模型独立工程（方案：球拍检测模型独立方案）。

对外只用三样东西：
* ``RacketObservation`` —— racket.v1 观测契约；
* ``RacketAdapter``    —— 检测后端 -> 观测；
* ``racket.fusion``    —— 球拍证据 -> HIT 候选打分（不判分）。
"""

from .schema import SCHEMA_VERSION, HitCandidate, ObservationLog, RacketObservation, SchemaError
from .tracker import Detection, RacketTracker, TrackerConfig, TrackState

__all__ = [
    "SCHEMA_VERSION",
    "RacketObservation",
    "HitCandidate",
    "ObservationLog",
    "SchemaError",
    "Detection",
    "RacketTracker",
    "TrackerConfig",
    "TrackState",
]

__version__ = "0.1.0"
