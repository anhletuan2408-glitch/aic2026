import unittest

import numpy as np

from hybrid_search import HybridConfig, reciprocal_rank_fusion


class HybridFusionTests(unittest.TestCase):
    def test_object_and_metadata_can_promote_a_base_candidate(self):
        config = HybridConfig(rrf_k=10, object_weight=1.0, metadata_weight=1.0)
        ids, scores = reciprocal_rank_fusion(
            [1, 2, 3], [3, 4], {1: "a", 2: "b", 3: "c", 4: "d"},
            {"c": 1, "a": 9, "b": 10, "d": 20}, config
        )
        self.assertEqual(ids[0], 3)
        self.assertTrue(np.all(scores[:-1] >= scores[1:]))

    def test_base_only_keeps_original_order(self):
        ids, _ = reciprocal_rank_fusion(
            [7, 8, 9], [], {7: "a", 8: "b", 9: "c"}, {}, HybridConfig()
        )
        self.assertEqual(ids, [7, 8, 9])
