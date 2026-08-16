import unittest

from review_queue import assess_results


class ReviewQueueTests(unittest.TestCase):
    def test_kis_low_margin_is_sent_to_review(self) -> None:
        decision = assess_results("kis", [
            {"video_id": "L21_V001", "score": 0.10},
            {"video_id": "L22_V001", "score": 0.099},
        ])
        self.assertTrue(decision.required)
        self.assertLess(decision.confidence, 0.74)

    def test_kis_strong_margin_and_video_support_can_auto_accept(self) -> None:
        rows = [
            {"video_id": "L21_V001", "score": score}
            for score in (0.10, 0.06, 0.05, 0.04, 0.03)
        ]
        decision = assess_results("kis", rows)
        self.assertFalse(decision.required)

    def test_qa_disagreement_is_prioritized(self) -> None:
        rows = [
            {"video_id": "L21_V001", "frame_idx": index, "answer": answer}
            for index, answer in enumerate(("red", "blue", "green", "yellow"))
        ]
        self.assertTrue(assess_results("qa", rows).required)

    def test_qa_numeric_aliases_count_as_consensus(self) -> None:
        answers = ("5", "Năm người", "five people", "5", "Năm người")
        rows = [
            {"video_id": "L21_V001", "frame_idx": index, "answer": answer}
            for index, answer in enumerate(answers)
        ]
        self.assertFalse(assess_results("qa", rows).required)


if __name__ == "__main__":
    unittest.main()
