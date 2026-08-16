import unittest

from evaluate_retrieval import aggregate, aggregate_by_round, score_ranking


class RetrievalEvaluationTests(unittest.TestCase):
    def test_official_kis_thresholds(self):
        rows = [
            {"video_id": "L01_V001", "frame_idx": 10},
            {"video_id": "L01_V002", "frame_idx": 20},
            {"video_id": "L01_V003", "frame_idx": 30},
        ]
        score = score_ranking(
            rows, {"video_id": "L01_V003", "start": 29, "end": 31}
        )
        self.assertEqual(score["r@1"], 0.0)
        self.assertEqual(score["r@5"], 1.0)
        self.assertEqual(score["first_relevant_rank"], 3.0)
        self.assertEqual(score["final_score"], 0.8)

    def test_aggregate(self):
        one = {"r@1": 1.0, "r@5": 1.0, "r@20": 1.0,
               "r@50": 1.0, "r@100": 1.0, "final_score": 1.0}
        zero = {key: 0.0 for key in one}
        self.assertEqual(aggregate([one, zero])["final_score"], 0.5)
    def test_aggregate_by_round(self):
        one = {"round": "r1", "r@1": 1.0, "r@5": 1.0, "r@20": 1.0,
               "r@50": 1.0, "r@100": 1.0, "final_score": 1.0}
        zero = {"round": "r2", "r@1": 0.0, "r@5": 0.0, "r@20": 0.0,
                "r@50": 0.0, "r@100": 0.0, "final_score": 0.0}
        metrics = aggregate_by_round([one, zero])
        self.assertEqual(metrics["r1"]["final_score"], 1.0)
        self.assertEqual(metrics["r2"]["r@100"], 0.0)
