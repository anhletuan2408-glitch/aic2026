from __future__ import annotations

import unittest

from trake_search import align_event_candidates, split_events


class TrakeSearchTests(unittest.TestCase):
    def test_split_numbered_lines(self) -> None:
        self.assertEqual(
            split_events("1. người mở cửa\n2. người bước vào\n3. người ngồi xuống"),
            ["người mở cửa", "người bước vào", "người ngồi xuống"],
        )

    def test_split_semicolon(self) -> None:
        self.assertEqual(split_events("mở cửa; bước vào; ngồi xuống"),
                         ["mở cửa", "bước vào", "ngồi xuống"])

    def test_alignment_requires_same_video_and_increasing_frames(self) -> None:
        rows = [
            [
                {"video_id": "L21_V001", "frame_idx": 100},
                {"video_id": "L21_V002", "frame_idx": 10},
            ],
            [
                {"video_id": "L21_V001", "frame_idx": 200},
                {"video_id": "L21_V002", "frame_idx": 5},
            ],
            [
                {"video_id": "L21_V001", "frame_idx": 300},
                {"video_id": "L21_V002", "frame_idx": 20},
            ],
        ]
        aligned = align_event_candidates(rows)
        self.assertEqual(aligned[0][1].video_id, "L21_V001")
        self.assertEqual(aligned[0][1].frame_ids, (100, 200, 300))
        self.assertTrue(all(
            a < b for _, answer in aligned
            for a, b in zip(answer.frame_ids, answer.frame_ids[1:])
        ))


if __name__ == "__main__":
    unittest.main()
