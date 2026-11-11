"""适配器单测：坐标/时间校验、Proposal 门控、观测日志。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from racket.adapter import AdapterConfig, ProposalGate, RacketAdapter
from racket.detectors.fake import EmptyDetector, StaticDetector
from racket.schema import ObservationLog, RacketObservation, read_observations
from racket.tracker import Detection

SIZE = (1920, 1080)


class TestAdapter(unittest.TestCase):
    def test_observations_are_valid_and_tracked(self):
        detector = StaticDetector({
            0: [Detection(bbox=(100, 100, 200, 200), confidence=0.9, side="A")],
            1: [Detection(bbox=(105, 100, 205, 200), confidence=0.9, side="A")],
        })
        adapter = RacketAdapter(detector, AdapterConfig(session_id="s01"))
        first = adapter.process_frame(None, 0, 0.0, SIZE)
        second = adapter.process_frame(None, 1, 0.033, SIZE)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].track_id, "racket_A_01")
        self.assertEqual(second[0].track_id, "racket_A_01")
        self.assertEqual(second[0].side, "A")
        self.assertAlmostEqual(second[0].confidence, 0.9)
        self.assertEqual(adapter.summary()["frames_processed"], 2)

    def test_low_confidence_dropped(self):
        detector = StaticDetector({
            0: [Detection(bbox=(100, 100, 200, 200), confidence=0.10, side="A")],
        })
        adapter = RacketAdapter(detector, AdapterConfig(session_id="s01", conf_thr=0.25))
        obs = adapter.process_frame(None, 0, 0.0, SIZE)
        self.assertEqual(obs, [])
        self.assertEqual(adapter.dropped_low_conf, 1)

    def test_out_of_image_box_is_clipped_not_crashed(self):
        detector = StaticDetector({
            0: [Detection(bbox=(-80, -40, 100, 120), confidence=0.9, side="A")],
        })
        adapter = RacketAdapter(detector, AdapterConfig(session_id="s01"))
        obs = adapter.process_frame(None, 0, 0.0, SIZE)
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].bbox_xyxy, (0.0, 0.0, 100.0, 120.0))

    def test_fully_outside_box_dropped(self):
        detector = StaticDetector({
            0: [Detection(bbox=(-100, -100, -10, -10), confidence=0.9, side="A")],
        })
        adapter = RacketAdapter(detector, AdapterConfig(session_id="s01"))
        self.assertEqual(adapter.process_frame(None, 0, 0.0, SIZE), [])
        self.assertEqual(adapter.dropped_outside, 1)

    def test_unknown_side_falls_back_to_geometry(self):
        detector = StaticDetector({
            0: [Detection(bbox=(1400, 100, 1500, 200), confidence=0.9, side="UNKNOWN", class_id=2)],
        })
        adapter = RacketAdapter(detector, AdapterConfig(session_id="s01"))
        obs = adapter.process_frame(None, 0, 0.0, SIZE)
        self.assertEqual(obs[0].side, "B")

    def test_proposal_gate_skips_frames(self):
        detector = StaticDetector({})
        gate = ProposalGate([(0.0, 0.5)])
        adapter = RacketAdapter(detector, AdapterConfig(session_id="s01"), gate=gate)
        self.assertEqual(adapter.process_frame(None, 0, 0.2, SIZE), [])
        adapter.process_frame(None, 1, 2.0, SIZE)
        self.assertEqual(adapter.frames_skipped, 1)
        self.assertEqual(detector.calls, 1)   # 被跳过的帧不调用模型

    def test_lost_frame_emitted_on_occlusion(self):
        detector = StaticDetector({
            0: [Detection(bbox=(100, 100, 200, 200), confidence=0.9, side="A")],
        })
        adapter = RacketAdapter(detector, AdapterConfig(session_id="s01"))
        adapter.process_frame(None, 0, 0.0, SIZE)
        obs = adapter.process_frame(None, 1, 0.033, SIZE)
        self.assertEqual(len(obs), 1)
        self.assertFalse(obs[0].visible)
        self.assertEqual(obs[0].status, "occluded")
        self.assertIsNone(obs[0].bbox_xyxy)

    def test_empty_detector_produces_nothing(self):
        adapter = RacketAdapter(EmptyDetector(), AdapterConfig(session_id="s01"))
        self.assertEqual(adapter.process_frame(None, 0, 0.0, SIZE), [])

    def test_log_written_and_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "obs.jsonl"
            detector = StaticDetector({
                0: [Detection(bbox=(100, 100, 200, 200), confidence=0.9, side="A")],
            })
            with ObservationLog(path, truncate=True) as log:
                adapter = RacketAdapter(detector, AdapterConfig(session_id="s01"), log=log)
                adapter.process_frame(None, 0, 0.0, SIZE)
                log.flush()
                items = list(read_observations(path))
            self.assertEqual(len(items), 1)
            self.assertIsInstance(items[0], RacketObservation)
            self.assertEqual(items[0].schema_version, "racket.v1")

    def test_max_boxes_limit(self):
        detector = StaticDetector({
            0: [
                Detection(bbox=(100, 100, 200, 200), confidence=0.9, side="A"),
                Detection(bbox=(300, 100, 400, 200), confidence=0.8, side="A"),
                Detection(bbox=(1400, 100, 1500, 200), confidence=0.7, side="B"),
            ],
        })
        adapter = RacketAdapter(detector, AdapterConfig(session_id="s01", max_boxes=2,
                                                        max_per_side=0))
        obs = adapter.process_frame(None, 0, 0.0, SIZE)
        self.assertEqual(len(obs), 2)
        self.assertAlmostEqual(obs[0].confidence, 0.9)

    def test_same_side_duplicate_dropped(self):
        """一个人只有一支球拍：同侧两个框只留置信度高的那个。"""
        detector = StaticDetector({
            0: [
                Detection(bbox=(100, 100, 200, 200), confidence=0.9, side="A"),
                Detection(bbox=(300, 100, 400, 200), confidence=0.55, side="A"),
                Detection(bbox=(1400, 100, 1500, 200), confidence=0.7, side="B"),
            ],
        })
        adapter = RacketAdapter(detector, AdapterConfig(session_id="s01"))
        obs = adapter.process_frame(None, 0, 0.0, SIZE)
        self.assertEqual(len(obs), 2)
        self.assertEqual(adapter.dropped_same_side, 1)
        self.assertAlmostEqual(obs[0].confidence, 0.9)
        self.assertEqual({o.side for o in obs}, {"A", "B"})

    def test_duplicate_dropped_can_be_disabled(self):
        detector = StaticDetector({
            0: [
                Detection(bbox=(100, 100, 200, 200), confidence=0.9, side="A"),
                Detection(bbox=(300, 100, 400, 200), confidence=0.8, side="A"),
            ],
        })
        adapter = RacketAdapter(detector, AdapterConfig(session_id="s01", max_per_side=0))
        self.assertEqual(len(adapter.process_frame(None, 0, 0.0, SIZE)), 2)
        self.assertEqual(adapter.dropped_same_side, 0)


if __name__ == "__main__":
    unittest.main()
