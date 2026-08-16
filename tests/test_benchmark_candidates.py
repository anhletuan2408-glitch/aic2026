from __future__ import annotations

import unittest

from benchmark_candidates import score_candidate_rows


class CandidateBenchmarkTests(unittest.TestCase):
    def test_scores_first_relevant_frame_at_official_thresholds(self) -> None:
        rows = [
            {"video_id": "L21_V001", "frame_idx": value}
            for value in (10, 20, 150, 250)
        ]
        truth = {"video_id": "L21_V001", "start": 100, "end": 200}
        score = score_candidate_rows(rows, truth)
        self.assertEqual(score["first_relevant_rank"], 3)
        self.assertEqual(score["r@1"], 0.0)
        self.assertEqual(score["r@5"], 1.0)
        self.assertEqual(score["final_score"], 0.8)
        self.assertEqual(score["first_same_video_rank"], 1)
        self.assertEqual(score["closest_frame_distance"], 50)

    def test_wrong_video_is_not_relevant(self) -> None:
        score = score_candidate_rows(
            [{"video_id": "L21_V002", "frame_idx": 150}],
            {"video_id": "L21_V001", "start": 100, "end": 200},
        )
        self.assertEqual(score["first_relevant_rank"], 0)
        self.assertEqual(score["final_score"], 0.0)
        self.assertEqual(score["first_same_video_rank"], 0)
        self.assertEqual(score["closest_frame_distance"], -1)


if __name__ == "__main__":
    unittest.main()
