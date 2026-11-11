"""数据集层单测：格式校验、按场次切分（防泄漏）、YOLO 导出。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from racket.dataset import (
    BoxLabel,
    DatasetError,
    FrameLabel,
    check_session_isolation,
    dataset_stats,
    export_yolo,
    load_frames,
    save_frames,
    split_sessions,
)


def make_image(path: Path, size=(640, 480)) -> str:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (40, 60, 90)).save(path)
    return str(path)


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.frames = []
        for session in ("s01", "s02", "s03", "s04", "s05"):
            for frame_id in range(2):
                image = make_image(self.root / "frames" / session / f"{frame_id:06d}.jpg")
                self.frames.append(FrameLabel(
                    session_id=session,
                    frame_id=frame_id,
                    timestamp_s=frame_id / 30.0,
                    image_path=image,
                    boxes=[BoxLabel(bbox_xyxy=(100, 100, 300, 300), side="A")],
                    image_size=(640, 480),
                ))

    def tearDown(self):
        self.tmp.cleanup()

    def test_stats(self):
        stats = dataset_stats(self.frames)
        self.assertEqual(stats["frames"], 10)
        self.assertEqual(stats["sessions"], 5)
        self.assertEqual(stats["boxes_per_side"]["A"], 10)

    def test_split_keeps_sessions_isolated(self):
        train, val, test = split_sessions(self.frames, seed=1)
        self.assertEqual(len(train) + len(val) + len(test), len(self.frames))
        train_sessions = {f.session_id for f in train}
        val_sessions = {f.session_id for f in val}
        test_sessions = {f.session_id for f in test}
        self.assertFalse(train_sessions & val_sessions)
        self.assertFalse(train_sessions & test_sessions)
        self.assertFalse(val_sessions & test_sessions)
        check_session_isolation(train, val, test)

    def test_split_requires_three_sessions(self):
        few = [f for f in self.frames if f.session_id in ("s01", "s02")]
        with self.assertRaises(DatasetError):
            split_sessions(few)

    def test_roundtrip_and_duplicate_detection(self):
        path = self.root / "labels.jsonl"
        save_frames(self.frames, path)
        loaded = load_frames(path)
        self.assertEqual(len(loaded), len(self.frames))
        self.assertEqual(loaded[0].boxes[0].bbox_xyxy, (100.0, 100.0, 300.0, 300.0))

        with open(path, "a", encoding="utf-8") as fh:
            fh.write(path.read_text(encoding="utf-8").splitlines()[0] + "\n")
        with self.assertRaises(DatasetError):
            load_frames(path)

    def test_bad_side_and_hard_tag_rejected(self):
        with self.assertRaises(DatasetError):
            BoxLabel(bbox_xyxy=(1, 1, 10, 10), side="left")
        with self.assertRaises(DatasetError):
            BoxLabel(bbox_xyxy=(1, 1, 10, 10), side="A", hard=("raining",))

    def test_export_yolo(self):
        train, val, test = split_sessions(self.frames, seed=1)
        dest = self.root / "yolo"
        info = export_yolo(self.frames, dest, splits={"train": train, "val": val, "test": test})
        self.assertEqual(info["counts"]["train"] + info["counts"]["val"] + info["counts"]["test"], 10)
        data_yaml = (dest / "data.yaml").read_text(encoding="utf-8")
        self.assertIn("racket_A", data_yaml)
        self.assertIn("train: images/train", data_yaml)

        label_files = list((dest / "labels" / "train").glob("*.txt"))
        self.assertTrue(label_files)
        parts = label_files[0].read_text(encoding="utf-8").split()
        self.assertEqual(parts[0], "0")                     # racket_A
        self.assertAlmostEqual(float(parts[1]), 200 / 640, places=4)
        self.assertAlmostEqual(float(parts[2]), 200 / 480, places=4)
        self.assertAlmostEqual(float(parts[3]), 200 / 640, places=4)
        self.assertAlmostEqual(float(parts[4]), 200 / 480, places=4)

    def test_export_reports_missing_frame(self):
        broken = FrameLabel(
            session_id="s01", frame_id=99, timestamp_s=1.0,
            image_path=str(self.root / "missing.jpg"),
            boxes=[], image_size=(640, 480),
        )
        with self.assertRaises(DatasetError):
            export_yolo([broken], self.root / "yolo2", splits={"train": [broken]})


if __name__ == "__main__":
    unittest.main()
