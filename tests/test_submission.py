from __future__ import annotations

import unittest

from submission import (
    FrameRange,
    KISAnswer,
    QAAnswer,
    TRAKEAnswer,
    final_score,
    kis_r_score,
    qa_r_score,
    qa_exact_r_score,
    trake_r_score,
    write_qa_submission,
    write_trake_submission,
)


class SubmissionScoringTests(unittest.TestCase):
    def test_kis_requires_video_and_frame_range(self) -> None:
        truth = FrameRange(500, 510)
        self.assertEqual(
            kis_r_score(KISAnswer("L01_V001", 505), "L01_V001", truth), 1.0
        )
        self.assertEqual(
            kis_r_score(KISAnswer("L01_V001", 600), "L01_V001", truth), 0.0
        )
        self.assertEqual(
            kis_r_score(KISAnswer("L02_V003", 505), "L01_V001", truth), 0.0
        )

    def test_qna_accepts_normalized_alias(self) -> None:
        score = qa_r_score(
            QAAnswer("L05_V005", 888, "  MÀU   XANH "),
            "L05_V005",
            FrameRange(800, 900),
            ["màu xanh", "blue"],
        )
        self.assertEqual(score, 1.0)
    def test_qna_exact_matches_official_description(self) -> None:
        answer = QAAnswer("L05_V005", 888, "Màu xanh")
        self.assertEqual(
            qa_exact_r_score(answer, "L05_V005", FrameRange(800, 900), "Màu xanh"),
            1.0,
        )
        self.assertEqual(
            qa_exact_r_score(answer, "L05_V005", FrameRange(800, 900), "màu xanh"),
            0.0,
        )

    def test_qa_and_trake_writers_have_no_header(self) -> None:
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            qa_path = Path(directory) / "q.csv"
            trake_path = Path(directory) / "t.csv"
            write_qa_submission(
                qa_path, [QAAnswer("L05_V005", 888, "Ba người, gồm hai nam")]
            )
            write_trake_submission(
                trake_path, [TRAKEAnswer("L10_V010", (100, 200, 300))]
            )
            self.assertEqual(
                qa_path.read_text(encoding="utf-8").strip(),
                'L05_V005,888,"Ba người, gồm hai nam"',
            )
            self.assertEqual(
                trake_path.read_text(encoding="utf-8").strip(),
                "L10_V010,100,200,300",
            )

    def test_trake_example_scores_three_of_four(self) -> None:
        ranges = [
            FrameRange(95, 105),
            FrameRange(145, 155),
            FrameRange(195, 205),
            FrameRange(245, 255),
        ]
        answer = TRAKEAnswer("L10_V010", (101, 156, 203, 251))
        self.assertEqual(trake_r_score(answer, "L10_V010", ranges), 0.75)

    def test_trake_wrong_video_is_zero(self) -> None:
        answer = TRAKEAnswer("L10_V011", (101, 150))
        self.assertEqual(
            trake_r_score(
                answer, "L10_V010", [FrameRange(95, 105), FrameRange(145, 155)]
            ),
            0.0,
        )

    def test_final_score_example(self) -> None:
        scores = [0.5, 0.1, 0.8, 0.2, 0.1] + [0.6] + [0.0] * 94
        top_scores, score = final_score(scores)
        self.assertEqual(
            top_scores, {1: 0.5, 5: 0.8, 20: 0.8, 50: 0.8, 100: 0.8}
        )
        self.assertAlmostEqual(score, 0.74)


if __name__ == "__main__":
    unittest.main()
