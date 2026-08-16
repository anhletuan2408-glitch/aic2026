import io
import unittest
import threading

import numpy as np
import torch
from PIL import Image

from rerank import RerankConfig, Siglip2Reranker, fuse_rerank_scores


class RerankFusionTests(unittest.TestCase):
    def test_half_precision_image_features_are_fused_with_float_text(self):
        class Inputs(dict):
            def to(self, _device):
                return self

        class Processor:
            def __call__(self, text=None, images=None, **_kwargs):
                if text is not None:
                    return Inputs(count=len(text))
                return Inputs(count=len(images))

        class Model:
            def get_text_features(self, count, **_kwargs):
                return torch.tensor([[3.0, 4.0]]).repeat(count, 1)

            def get_image_features(self, count, **_kwargs):
                return torch.tensor(
                    [[3.0, 4.0]], dtype=torch.float16
                ).repeat(count, 1)

        payload = io.BytesIO()
        Image.new("RGB", (2, 2), "red").save(payload, format="JPEG")
        frames = type(
            "Frames", (), {"get_bytes": lambda *_args: payload.getvalue()}
        )()
        reranker = Siglip2Reranker.__new__(Siglip2Reranker)
        reranker.device = "cpu"
        reranker.processor = Processor()
        reranker.model = Model()
        reranker.keyframes = frames
        reranker.config = RerankConfig(batch_size=1)
        reranker._lock = threading.Lock()

        output = reranker.rerank(
            "red square",
            [{"rank": 1, "score": .5, "video_id": "V", "keyframe_no": 1}],
        )
        self.assertAlmostEqual(output[0]["siglip2_score"], 1.0, places=3)

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
