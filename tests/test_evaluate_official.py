from __future__ import annotations

import unittest

from evaluate import evaluate_qna


class OfficialEvaluationTests(unittest.TestCase):
    def test_qna_is_exact_not_normalized(self) -> None:
        truth = {
            "video_id": "L05_V005",
            "start": 100,
            "end": 200,
            "answers": ["Màu xanh", "Blue"],
        }
        self.assertEqual(
            evaluate_qna([["L05_V005", "150", "Màu xanh"]], truth), [1.0]
        )
        self.assertEqual(
            evaluate_qna([["L05_V005", "150", "màu xanh"]], truth), [0.0]
        )


if __name__ == "__main__":
    unittest.main()
