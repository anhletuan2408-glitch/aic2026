from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from trake_search import (
    add_video_conditioned_candidates, align_event_candidates,
    fuse_event_rankings, joint_video_candidates, split_events,
)


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


    def test_widest_gap_variant_follows_best_path_for_priority_video(self) -> None:
        rows = [
            [
                {"video_id": "A", "frame_idx": 100, "pts_time": 0.0, "score": .9},
                {"video_id": "B", "frame_idx": 100, "pts_time": 0.0, "score": .4},
            ],
            [
                {"video_id": "A", "frame_idx": 200, "pts_time": 10.0, "score": .95},
                {"video_id": "A", "frame_idx": 500, "pts_time": 400.0, "score": .7},
                {"video_id": "B", "frame_idx": 200, "pts_time": 200.0, "score": .4},
            ],
            [
                {"video_id": "A", "frame_idx": 300, "pts_time": 20.0, "score": .95},
                {"video_id": "A", "frame_idx": 900, "pts_time": 800.0, "score": .7},
                {"video_id": "B", "frame_idx": 300, "pts_time": 400.0, "score": .4},
            ],
        ]
        aligned = align_event_candidates(
            rows, gap_profiles=(0.0, 100.0), priority_videos=2,
        )
        self.assertEqual(aligned[0][1].frame_ids, (100, 200, 300))
        self.assertEqual(aligned[1][1].frame_ids, (100, 500, 900))


    def test_conditioned_pass_searches_all_frames_inside_joint_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "metadata.sqlite3"
            connection = sqlite3.connect(metadata)
            connection.execute(
                "CREATE TABLE keyframes("
                "global_id INTEGER,video_id TEXT,frame_idx INTEGER,pts_time REAL)"
            )
            connection.executemany(
                "INSERT INTO keyframes VALUES(?,?,?,?)",
                [(0, "A", 100, 1.0), (1, "A", 200, 2.0), (2, "A", 300, 3.0)],
            )
            connection.commit()
            connection.close()
            frame_vectors = np.asarray(
                [[1.0, 0.0], [0.0, 1.0], [.7, .7]], dtype=np.float32
            )
            engine = SimpleNamespace(
                metadata_path=metadata,
                index=SimpleNamespace(
                    reconstruct_batch=lambda ids: frame_vectors[ids]
                ),
            )
            rows = [[], []]
            add_video_conditioned_candidates(
                engine,
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                rows, ["A"], per_video=1,
            )
            self.assertEqual(rows[0][0]["frame_idx"], 100)
            self.assertEqual(rows[1][0]["frame_idx"], 200)

            rank_rows = [
                [{"video_id": "A", "frame_idx": 50, "pts_time": 0.0, "score": -2.0}],
                [{"video_id": "A", "frame_idx": 60, "pts_time": 0.0, "score": -2.0}],
            ]
            add_video_conditioned_candidates(
                engine,
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                rank_rows, ["A"], per_video=1,
            )
            self.assertEqual(rank_rows[0][0]["frame_idx"], 50)
            self.assertTrue(all(
                float(row["score"]) == -2.0 for rows in rank_rows for row in rows
            ))

    def test_event_fusion_preserves_legacy_scores_and_adds_siglip_recall(self) -> None:
        base_ids = np.asarray([10, 20, 30], dtype=np.int64)
        base_scores = np.asarray([.9, .8, .7], dtype=np.float32)
        legacy = fuse_event_rankings(base_ids, base_scores)
        self.assertEqual([global_id for global_id, _ in legacy], [10, 20, 30])
        np.testing.assert_allclose([score for _, score in legacy], base_scores)

        fused = fuse_event_rankings(
            base_ids, base_scores, np.asarray([30, 40, 10]), limit=4
        )
        self.assertEqual({global_id for global_id, _ in fused}, {10, 20, 30, 40})
        self.assertTrue(all(score == -2.0 for _, score in fused))

    def test_joint_video_candidates_require_all_events(self) -> None:
        rows = [
            [
                {"video_id": "A", "score": .9},
                {"video_id": "B", "score": .8},
                {"video_id": "only-first", "score": 1.0},
            ],
            [
                {"video_id": "B", "score": .95},
                {"video_id": "A", "score": .7},
            ],
        ]
        self.assertEqual(joint_video_candidates(rows, limit=2), ["B", "A"])

if __name__ == "__main__":
    unittest.main()
