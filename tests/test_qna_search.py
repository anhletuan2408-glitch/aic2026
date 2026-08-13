from __future__ import annotations

import unittest

from qna_search import (
    clean_answer, fuse_qa_candidate_rows, rank_qa_answers,
)


class QnaSearchTests(unittest.TestCase):

    def test_clean_answer_removes_wrapper(self) -> None:
        self.assertEqual(clean_answer('Answer: "Five people."'), "Five people")

    def test_candidate_fusion_adds_original_question_recall(self) -> None:
        primary = [
            {"video_id": "A", "frame_idx": 10},
            {"video_id": "B", "frame_idx": 20},
        ]
        secondary = [
            {"video_id": "C", "frame_idx": 30},
            {"video_id": "A", "frame_idx": 10},
        ]
        fused = fuse_qa_candidate_rows(primary, secondary)
        self.assertEqual((fused[0]["video_id"], fused[0]["frame_idx"]), ("A", 10))
        self.assertIn(
            ("C", 30),
            [(row["video_id"], row["frame_idx"]) for row in fused],
        )
        self.assertEqual([row["rank"] for row in fused], [1, 2, 3])

    def test_candidate_fusion_protects_quality_top_ten(self) -> None:
        primary = [
            {"video_id": f"P{index}", "frame_idx": index}
            for index in range(12)
        ]
        secondary = [
            {"video_id": f"S{index}", "frame_idx": index}
            for index in range(100)
        ]
        fused = fuse_qa_candidate_rows(primary, secondary, limit=20)
        self.assertEqual(
            [(row["video_id"], row["frame_idx"]) for row in fused[:10]],
            [(row["video_id"], row["frame_idx"]) for row in primary[:10]],
        )
    def test_consensus_answers_fill_ranked_frames_without_header(self) -> None:
        rows = [
            {"video_id": "L21_V001", "frame_idx": 100},
            {"video_id": "L21_V002", "frame_idx": 200},
            {"video_id": "L21_V003", "frame_idx": 300},
        ]
        answers = rank_qa_answers(
            rows, [(0, "Five"), (1, "Five"), (2, "Four")]
        )
        self.assertEqual(
            (answers[0].video_id, answers[0].frame_id, answers[0].answer),
            ("L21_V001", 100, "Five"),
        )
        self.assertIn(
            ("L21_V003", 300, "Five"),
            [(item.video_id, item.frame_id, item.answer) for item in answers],
        )


if __name__ == "__main__":
    unittest.main()
