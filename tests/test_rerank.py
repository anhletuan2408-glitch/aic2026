import unittest

import numpy as np

from rerank import RerankConfig, fuse_rerank_scores


class RerankFusionTests(unittest.TestCase):
    def test_siglip_can_promote_a_relevant_candidate(self):
        rows = [
            {"rank": 1, "score": 0.9, "video_id": "a"},
            {"rank": 2, "score": 0.8, "video_id": "b"},
            {"rank": 3, "score": 0.7, "video_id": "c"},
        ]
        output = fuse_rerank_scores(
            rows, np.asarray([0.1, 0.2, 0.9]), RerankConfig(weight=2.0)
        )
        self.assertEqual(output[0]["video_id"], "c")
        self.assertEqual([row["rank"] for row in output], [1, 2, 3])
        self.assertIn("pre_rerank_score", output[0])
