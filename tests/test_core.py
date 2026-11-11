"""球拍检测模块单测：契约、几何、跟踪。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from racket.geometry import center_distance, clip_bbox, iou, side_from_bbox, side_from_x
from racket.schema import (
    SCHEMA_VERSION,
    HitCandidate,
    ObservationLog,
    RacketObservation,
    SchemaError,
    read_observations,
)
from racket.tracker import Detection, RacketTracker, TrackerConfig


class TestSchema(unittest.TestCase):
    def test_valid_observation_roundtrip(self):
        obs = RacketObservation(
            session_id="s01", frame_id=900, timestamp_s=15.0,
            side="A", bbox_xyxy=(500, 400, 650, 620), confidence=0.88,
            track_id="racket_A_01",
        )
        data = obs.to_dict()
        self.assertEqual(data["schema_version"], SCHEMA_VERSION)
        self.assertEqual(data["center_px"], [575.0, 510.0])
        self.assertTrue(data["visible"])
        again = RacketObservation.from_dict(data)
        self.assertEqual(again, obs)

    def test_visible_requires_bbox_and_confidence(self):
        with self.assertRaises(SchemaError):
            RacketObservation(session_id="s01", frame_id=1, timestamp_s=0.1, visible=True)
        with self.assertRaises(SchemaError):
            RacketObservation(
                session_id="s01", frame_id=1, timestamp_s=0.1,
                bbox_xyxy=(1, 1, 10, 10), confidence=None,
            )

    def test_invisible_must_not_carry_coordinates(self):
        with self.assertRaises(SchemaError):
            RacketObservation(
                session_id="s01", frame_id=1, timestamp_s=0.1, visible=False,
                bbox_xyxy=(1, 1, 10, 10),
            )
        obs = RacketObservation.lost("s01", 1, 0.1)
        self.assertIsNone(obs.bbox_xyxy)
        self.assertEqual(obs.status, "lost")

    def test_bbox_outside_image_rejected(self):
        with self.assertRaises(SchemaError):
            RacketObservation(
                session_id="s01", frame_id=1, timestamp_s=0.1,
                bbox_xyxy=(1900, 100, 2000, 200), confidence=0.5,
                image_size=(1920, 1080),
            )

    def test_confidence_range_enforced(self):
        with self.assertRaises(SchemaError):
            RacketObservation(
                session_id="s01", frame_id=1, timestamp_s=0.1,
                bbox_xyxy=(1, 1, 10, 10), confidence=1.5,
            )

    def test_negative_frame_id_and_bad_side(self):
        with self.assertRaises(SchemaError):
            RacketObservation(session_id="s01", frame_id=-1, timestamp_s=0.1)
        with self.assertRaises(SchemaError):
            RacketObservation(session_id="s01", frame_id=1, timestamp_s=0.1, side="left")

    def test_observation_log_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "obs.jsonl"
            with ObservationLog(path, truncate=True) as log:
                log.write(RacketObservation(
                    session_id="s01", frame_id=1, timestamp_s=0.033,
                    side="B", bbox_xyxy=(10, 10, 60, 80), confidence=0.7,
                ))
                log.write(RacketObservation.lost("s01", 2, 0.066))
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("{坏行}\n")
            items = list(read_observations(path))
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0].side, "B")
            self.assertFalse(items[1].visible)

    def test_hit_candidate_validation(self):
        cand = HitCandidate(event_id="s01:hit:0042", timestamp_s=15.02, side="A",
                            quality_score=0.82, decision="candidate")
        self.assertEqual(cand.to_dict()["quality_score"], 0.82)
        with self.assertRaises(SchemaError):
            HitCandidate(event_id="x", timestamp_s=1.0, quality_score=1.4)


class TestGeometry(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (10, 0, 20, 10)), 0.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3, places=5)

    def test_side_from_x(self):
        self.assertEqual(side_from_x(100, 1920), "A")
        self.assertEqual(side_from_x(1800, 1920), "B")
        self.assertEqual(side_from_x(960, 1920), "UNKNOWN")   # 压网线不猜
        self.assertEqual(side_from_x(985, 1920), "UNKNOWN")   # 死区内
        self.assertEqual(side_from_x(1000, 1920), "B")

    def test_side_from_bbox_and_clip(self):
        self.assertEqual(side_from_bbox((100, 100, 200, 200), (1920, 1080)), "A")
        clipped = clip_bbox((-50, -10, 100, 50), 1920, 1080)
        self.assertEqual(clipped, (0.0, 0.0, 100.0, 50.0))
        self.assertGreater(center_distance((0, 0, 10, 10), (30, 40, 40, 50)), 0)


class TestTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = RacketTracker(TrackerConfig(iou_threshold=0.2, max_age_frames=3))

    def test_stable_id_across_frames(self):
        for frame in range(3):
            states = self.tracker.update(
                [Detection(bbox=(100, 100, 200, 200), confidence=0.9, side="A")],
                frame, frame * 0.033,
            )
        visible = [s for s in states if s.status == "ok"]
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0].track_id, "racket_A_01")

    def test_two_sides_get_separate_tracks(self):
        states = self.tracker.update([
            Detection(bbox=(100, 100, 200, 200), confidence=0.9, side="A"),
            Detection(bbox=(1400, 100, 1500, 200), confidence=0.8, side="B"),
        ], 0, 0.0)
        ids = sorted(s.track_id for s in states)
        self.assertEqual(ids, ["racket_A_01", "racket_B_01"])

    def test_occluded_then_closed(self):
        self.tracker.update([Detection(bbox=(100, 100, 200, 200), confidence=0.9, side="A")], 0, 0.0)
        states = self.tracker.update([], 1, 0.033)
        self.assertEqual([s.status for s in states], ["occluded"])
        self.assertIsNone(states[0].bbox)
        for frame in range(2, 8):
            states = self.tracker.update([], frame, frame * 0.033)
        self.assertEqual(states, [])
        self.assertEqual(self.tracker.stats()["tracks_closed"], 1)

    def test_velocity_and_direction(self):
        self.tracker.update([Detection(bbox=(100, 100, 200, 200), confidence=0.9, side="A")], 0, 0.0)
        states = self.tracker.update(
            [Detection(bbox=(160, 100, 260, 200), confidence=0.9, side="A")], 1, 0.1,
        )
        state = [s for s in states if s.status == "ok"][0]
        self.assertGreater(state.velocity_px_s[0], 0)
        self.assertGreater(state.speed_px_s, 0)
        self.assertEqual(state.direction_deg, 0.0)

    def test_side_switch_not_allowed_by_default(self):
        self.tracker.update([Detection(bbox=(100, 100, 200, 200), confidence=0.9, side="A")], 0, 0.0)
        states = self.tracker.update(
            [Detection(bbox=(1400, 100, 1500, 200), confidence=0.9, side="B")], 1, 0.033,
        )
        self.assertEqual(sorted(s.track_id for s in states), ["racket_A_01", "racket_B_01"])

    def test_gap_stats_reported(self):
        self.tracker.update([Detection(bbox=(100, 100, 200, 200), confidence=0.9, side="A")], 0, 0.0)
        self.tracker.update([], 1, 0.033)
        self.tracker.update([Detection(bbox=(100, 100, 200, 200), confidence=0.9, side="A")], 2, 0.066)
        stats = self.tracker.stats()
        self.assertEqual(stats["gap_events"], 1)
        self.assertGreater(stats["gap_frames"], 0)


if __name__ == "__main__":
    unittest.main()
