import unittest

import numpy as np

from hybrid_search import (
    HybridConfig, HybridSignals, object_concept_groups, reciprocal_rank_fusion,
)


class HybridFusionTests(unittest.TestCase):
    def test_object_conjunction_requires_person_and_motorcycle(self):
        signals = HybridSignals.__new__(HybridSignals)
        signals.labels = ["Woman", "Person", "Motorcycle", "Cat"]
        signals._label_by_name = {
            label.casefold(): index for index, label in enumerate(signals.labels)
        }
        signals.frame_count = 4
        signals.offsets = np.asarray([0, 2, 5, 7, 8])
        signals.frame_ids = np.asarray([0, 1, 0, 1, 2, 0, 2, 3])
        signals.scores = np.asarray([.9, .8, .85, .8, .7, .95, .6, .99])
        ranked, labels = signals.object_conjunction_ranking(
            "một người phụ nữ chạy xe máy", HybridConfig()
        )
        self.assertEqual(ranked[:2], [0, 2])
        self.assertNotIn(1, ranked)
        self.assertEqual(labels, ["Woman/Person", "Motorcycle"])
        self.assertEqual(object_concept_groups("một con mèo màu đen"), [("Cat",)])
        self.assertEqual(
            object_concept_groups("người đàn ông ngồi xuống ghế"),
            [("Man", "Person"), ("Chair", "Couch", "Bench")],
        )

    def test_object_and_metadata_can_promote_a_base_candidate(self):
        config = HybridConfig(rrf_k=10, object_weight=1.0, metadata_weight=1.0)
        ids, scores = reciprocal_rank_fusion(
            [1, 2, 3], [3, 4], {1: "a", 2: "b", 3: "c", 4: "d"},
            {"c": 1, "a": 9, "b": 10, "d": 20}, config
        )
        self.assertEqual(ids[0], 3)
        self.assertTrue(np.all(scores[:-1] >= scores[1:]))

    def test_ocr_exact_text_can_promote_a_candidate(self):
        config = HybridConfig(rrf_k=10, ocr_weight=2.0)
        ids, _ = reciprocal_rank_fusion(
            [1, 2, 3], [], {1:"a",2:"b",3:"c"}, {}, config,
            ocr_ids=[3],
        )
        self.assertEqual(ids[0], 3)
    def test_base_only_keeps_original_order(self):
        ids, _ = reciprocal_rank_fusion(
            [7, 8, 9], [], {7: "a", 8: "b", 9: "c"}, {}, HybridConfig()
        )
        self.assertEqual(ids, [7, 8, 9])
