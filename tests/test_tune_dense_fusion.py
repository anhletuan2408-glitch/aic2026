from __future__ import annotations

import unittest

from tune_dense_fusion import (
    DenseCase, first_target_rank, official_score, summarize,
)


class DenseFusionTuningTests(unittest.TestCase):
    def test_official_score_uses_competition_thresholds(self) -> None:
        self.assertEqual(official_score(1), 1.0)
        self.assertEqual(official_score(3), 0.8)
        self.assertEqual(official_score(21), 0.4)
        self.assertEqual(official_score(None), 0.0)

    def test_weight_sweep_can_promote_siglip_target(self) -> None:
        case = DenseCase(
            "q1", "kis", {3},
            clip_ids=[1, 2, 3],
            siglip_ids=[3, 4, 1],
        )
        report = summarize([case], [0.0, 2.0], limit=4)
        self.assertEqual(report["0.0"]["ranks"]["q1"], 3)
        self.assertEqual(report["2.0"]["ranks"]["q1"], 1)
        self.assertGreater(
            report["2.0"]["final_score"], report["0.0"]["final_score"]
        )
        self.assertEqual(first_target_rank([1, 2], {9}), None)


if __name__ == "__main__":
    unittest.main()
