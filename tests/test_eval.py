"""评估与融合单测：指标方向性必须对，不能"跑通但算错"。"""

from __future__ import annotations

import unittest

from racket.dataset import BoxLabel, FrameLabel, HitLabel
from racket.eval import (
    PredBox,
    PredFrame,
    evaluate_detection,
    evaluate_hit_windows,
    evaluate_proposals,
    format_report,
)
from racket.fusion import (
    HitWeights,
    build_candidate,
    direction_compatibility,
    is_suspect_hit,
    racket_ball_relation,
    score_hit,
    temporal_factor,
)
from racket.geometry import bbox_center


def gt_frame(side="A", frame_id=0, timestamp_s=0.0, bbox=(100, 100, 200, 200)):
    return FrameLabel(
        session_id="s01", frame_id=frame_id, timestamp_s=timestamp_s,
        image_path="x.jpg",
        boxes=[BoxLabel(bbox_xyxy=bbox, side=side)],
        image_size=(1920, 1080),
    )


class TestDetectionEval(unittest.TestCase):
    def test_perfect_prediction(self):
        frame = gt_frame()
        preds = [PredFrame("s01", 0, 0.0, [PredBox((100, 100, 200, 200), 0.9, "A")])]
        result = evaluate_detection([frame], preds)
        self.assertAlmostEqual(result["mAP50"], 1.0, places=4)
        self.assertAlmostEqual(result["side_accuracy"], 1.0)
        self.assertAlmostEqual(result["precision_recall"]["0.25"]["recall"], 1.0)

    def test_missed_detection(self):
        frame = gt_frame()
        result = evaluate_detection([frame], [PredFrame("s01", 0, 0.0, [])])
        self.assertAlmostEqual(result["mAP50"], 0.0)
        self.assertEqual(result["frame_recall"], 0.0)

    def test_wrong_side_kills_side_accuracy(self):
        frame = gt_frame(side="A")
        preds = [PredFrame("s01", 0, 0.0, [PredBox((100, 100, 200, 200), 0.9, "B")])]
        result = evaluate_detection([frame], preds)
        self.assertAlmostEqual(result["side_accuracy"], 0.0)

    def test_unknown_side_counted_separately(self):
        frame = gt_frame(side="A")
        preds = [PredFrame("s01", 0, 0.0, [PredBox((100, 100, 200, 200), 0.9, "UNKNOWN")])]
        result = evaluate_detection([frame], preds)
        self.assertAlmostEqual(result["side_unknown_rate"], 1.0)

    def test_hit_window_recall(self):
        hits = [HitLabel("s01", "h1", 1.0, 1.1, "A")]
        preds = [
            PredFrame("s01", 30, 1.05, [PredBox((100, 100, 200, 200), 0.8, "A")]),
            PredFrame("s01", 60, 2.00, [PredBox((100, 100, 200, 200), 0.8, "A")]),
        ]
        result = evaluate_hit_windows(hits, preds, window_s=0.2)
        self.assertAlmostEqual(result["hit_recall"], 1.0)
        self.assertAlmostEqual(result["hit_recall_correct_side"], 1.0)

    def test_hit_window_miss_and_wrong_side(self):
        hits = [HitLabel("s01", "h1", 1.0, 1.1, "A")]
        preds = [PredFrame("s01", 30, 1.05, [PredBox((100, 100, 200, 200), 0.8, "B")])]
        result = evaluate_hit_windows(hits, preds)
        self.assertAlmostEqual(result["hit_recall"], 1.0)          # 有框
        self.assertAlmostEqual(result["hit_recall_correct_side"], 0.0)  # 侧别错

    def test_proposals(self):
        preds = [PredFrame("s01", 1, 0.5, [PredBox((1, 1, 50, 50), 0.9, "A")])]
        result = evaluate_proposals([(0.0, 1.0), (5.0, 6.0)], preds)
        self.assertAlmostEqual(result["proposal_recall"], 0.5)

    def test_report_renders(self):
        frame = gt_frame()
        preds = [PredFrame("s01", 0, 0.0, [PredBox((100, 100, 200, 200), 0.9, "A")])]
        report = format_report(
            evaluate_detection([frame], preds),
            hits=evaluate_hit_windows([HitLabel("s01", "h1", 0.0, 0.1, "A")], preds),
        )
        self.assertIn("mAP@0.5", report)
        self.assertIn("击球时刻", report)


class TestFusion(unittest.TestCase):
    def test_temporal_factor(self):
        self.assertAlmostEqual(temporal_factor(0.0), 1.0)
        self.assertAlmostEqual(temporal_factor(0.2), 0.0)
        self.assertAlmostEqual(temporal_factor(0.4), 0.0)

    def test_relation_prefers_close_and_simultaneous(self):
        bbox = (100, 100, 200, 200)
        close = racket_ball_relation(bbox, bbox_center(bbox), 0.01)
        far = racket_ball_relation(bbox, (900, 900), 0.01)
        late = racket_ball_relation(bbox, bbox_center(bbox), 0.5)
        self.assertGreater(close, 0.8)
        self.assertLess(far, 0.3)
        self.assertAlmostEqual(late, 0.0)

    def test_direction_compatibility(self):
        aligned = direction_compatibility((100, 0), (0, 0), (100, 0))
        opposite = direction_compatibility((-100, 0), (0, 0), (100, 0))
        self.assertGreater(aligned, 0.99)
        self.assertLess(opposite, 0.01)
        self.assertAlmostEqual(direction_compatibility((0, 0), (0, 0), (1, 1)), 0.0)

    def test_score_hit_with_and_without_racket(self):
        full, evidence = score_hit(
            trajectory_change=1.0, racket_bbox=(100, 100, 200, 200),
            ball_center=(150, 150), dt_s=0.0, ball_speed_change=1.0,
            racket_velocity=(10, 0), ball_velocity_before=(-5, 0), ball_velocity_after=(5, 0),
        )
        self.assertGreater(full, 0.9)
        self.assertIn("racket_ball_close", evidence)

        degraded, evidence2 = score_hit(
            trajectory_change=1.0, racket_bbox=None, ball_center=None,
            dt_s=0.0, ball_speed_change=1.0,
        )
        self.assertIn("degraded_no_racket", evidence2)
        self.assertGreater(degraded, 0.0)

    def test_weights_normalize(self):
        w = HitWeights(1, 1, 1, 1).normalized()
        self.assertAlmostEqual(w.total(), 1.0)
        self.assertAlmostEqual(w.trajectory_change, 0.25)

    def test_build_candidate_decision(self):
        high = build_candidate(session_id="s01", sequence=42, timestamp_s=15.02,
                               side="A", score=0.9, evidence=["racket_ball_close"])
        self.assertEqual(high.decision, "candidate")
        self.assertEqual(high.event_id, "s01:hit:0042")
        low = build_candidate(session_id="s01", sequence=43, timestamp_s=16.0,
                              side="B", score=0.2, evidence=[])
        self.assertEqual(low.decision, "pending")

    def test_suspect_hit_rule(self):
        self.assertTrue(is_suspect_hit(
            hit_y_px=500, table_top_y_px=400, table_bottom_y_px=650,
            bounce_times_s=[5.05], hit_timestamp_s=5.0,
        ))
        self.assertFalse(is_suspect_hit(
            hit_y_px=200, table_top_y_px=400, table_bottom_y_px=650,
            bounce_times_s=[5.05], hit_timestamp_s=5.0,
        ))
        self.assertFalse(is_suspect_hit(
            hit_y_px=500, table_top_y_px=400, table_bottom_y_px=650,
            bounce_times_s=[9.0], hit_timestamp_s=5.0,
        ))


if __name__ == "__main__":
    unittest.main()
