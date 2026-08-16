import unittest
import threading

import numpy as np
import torch

from rerank import RerankConfig, Siglip2Reranker, fuse_rerank_scores


class RerankFusionTests(unittest.TestCase):
    def test_text_encoder_returns_normalized_float32_vectors(self):
        class Inputs(dict):
            def to(self, _device):
                return self

        class Processor:
            def __call__(self, text, **_kwargs):
                return Inputs(count=len(text))

        class Model:
            def get_text_features(self, count, **_kwargs):
                return torch.tensor([[3.0, 4.0]]).repeat(count, 1)

        reranker = Siglip2Reranker.__new__(Siglip2Reranker)
        reranker.device = "cpu"
        reranker.processor = Processor()
        reranker.model = Model()
        reranker._lock = threading.Lock()
        vector = reranker.encode_text("motorcycle")
        vectors = reranker.encode_text_many(["motorcycle", "car"])
        self.assertEqual(vector.dtype, np.float32)
        self.assertEqual(vectors.shape, (2, 2))
        np.testing.assert_allclose(vector, [[0.6, 0.8]], atol=1e-6)
        np.testing.assert_allclose(vectors, [[0.6, 0.8], [0.6, 0.8]], atol=1e-6)
        with self.assertRaises(ValueError):
            reranker.encode_text_many([])

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
