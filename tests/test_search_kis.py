from __future__ import annotations

import unittest

import numpy as np

from search_kis import select_candidates


class CandidateSelectionTests(unittest.TestCase):
    def test_round_robin_diversifies_videos_and_time(self) -> None:
        metadata = {
            1: {
                "video_id": "L21_V001",
                "keyframe_no": 1,
                "frame_idx": 0,
                "pts_time": 0.0,
                "title": "A",
            },
            2: {
                "video_id": "L21_V001",
                "keyframe_no": 2,
                "frame_idx": 30,
                "pts_time": 1.0,
                "title": "A",
            },
            3: {
                "video_id": "L21_V002",
                "keyframe_no": 1,
                "frame_idx": 0,
                "pts_time": 0.0,
                "title": "B",
            },
            4: {
                "video_id": "L21_V001",
                "keyframe_no": 3,
                "frame_idx": 90,
                "pts_time": 3.0,
                "title": "A",
            },
        }
        selected = select_candidates(
            [1, 2, 3, 4],
            np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
            metadata,
            top_k=3,
            per_video_limit=2,
            min_time_gap=2.0,
        )
        self.assertEqual(
            [(row["video_id"], row["frame_idx"]) for row in selected],
            [("L21_V001", 0), ("L21_V002", 0), ("L21_V001", 90)],
        )
        self.assertEqual([row["rank"] for row in selected], [1, 2, 3])

    def test_negative_time_gap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_candidates([], np.array([], dtype=np.float32), {}, 1, 1, -1.0)


if __name__ == "__main__":
    unittest.main()
