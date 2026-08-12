from __future__ import annotations

import unittest

from qna_search import clean_answer, rank_qa_answers


class QnaSearchTests(unittest.TestCase):
    def test_clean_answer_removes_wrapper(self) -> None:
        self.assertEqual(clean_answer('Answer: "Five people."'), "Five people")

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
