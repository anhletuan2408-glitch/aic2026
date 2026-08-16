from __future__ import annotations

import unittest

from retrieval_enhancements import (
    expand_query, fuse_query_rankings, qa_answer_hypothesis_queries,
    normalize_visual_query, qa_retrieval_query, temporal_event_variants,
    temporal_neighbor_ranking,
)


class RetrievalEnhancementTests(unittest.TestCase):
    def test_temporal_event_variant_preserves_chair_relation(self) -> None:
        self.assertEqual(
            temporal_event_variants("một người đàn ông đứng trước ghế"),
            ["một người đàn ông đứng trước ghế",
             "a man standing in front of a chair"],
        )
        self.assertIn(
            "a man sitting down on a chair",
            temporal_event_variants("người đàn ông ngồi xuống ghế"),
        )

    def test_repairs_common_vietnamese_ime_errors_without_splitting_relation(self) -> None:
        self.assertEqual(
            normalize_visual_query("momojt nguoi phu nu chay xe may"),
            "một người phụ nữ chạy xe máy",
        )
        variants = expand_query("momojt nguoi phu nu chay xe may")
        self.assertEqual(variants[0], "một người phụ nữ chạy xe máy")
        self.assertIn("a woman riding a motorcycle", variants)
        self.assertNotIn("xe máy", variants)

    def test_qa_answer_hypotheses_cover_common_answer_types(self) -> None:
        colors = qa_answer_hypothesis_queries("Chiếc áo có màu gì?")
        counts = qa_answer_hypothesis_queries("Có bao nhiêu người trước bảng?")
        sports = qa_answer_hypothesis_queries("Họ thi đấu môn gì?")
        self.assertTrue(any("màu đỏ red" in item for item in colors))
        self.assertTrue(any("một người one person" in item for item in counts))
        self.assertTrue(any("đua xe đạp cycling" in item for item in sports))
    def test_short_query_is_not_diluted(self) -> None:
        self.assertEqual(expand_query("xe máy"), ["xe máy"])

    def test_long_query_gets_content_variant(self) -> None:
        variants = expand_query(
            "Một người phụ nữ đang chạy xe máy trên đường đông người"
        )
        self.assertEqual(
            variants[0],
            "Một người phụ nữ đang chạy xe máy trên đường đông người",
        )
        self.assertIn("người phụ nữ chạy xe máy đường đông người", variants)

    def test_qa_retrieval_query_removes_answer_slot_not_scene(self) -> None:
        self.assertEqual(
            qa_retrieval_query(
                "C\u00f3 bao nhi\u00eau ng\u01b0\u1eddi ch\u00ednh \u0111ang "
                "\u0111\u1ee9ng tr\u01b0\u1edbc b\u1ea3ng tr\u1eafng v\u00e0 vi\u1ebft b\u00e0i?"
            ),
            "ng\u01b0\u1eddi ch\u00ednh \u0111ang \u0111\u1ee9ng tr\u01b0\u1edbc "
            "b\u1ea3ng tr\u1eafng v\u00e0 vi\u1ebft b\u00e0i",
        )
        self.assertEqual(
            qa_retrieval_query(
                "Tr\u00ean m\u00e0n h\u00ecnh n\u1ec1n \u0111en, d\u1ea5u X c\u00f3 m\u00e0u g\u00ec?"
            ),
            "Tr\u00ean m\u00e0n h\u00ecnh n\u1ec1n \u0111en, d\u1ea5u X",
        )

    def test_query_rrf_keeps_original_dominant(self) -> None:
        ids, _ = fuse_query_rankings([[1, 2, 3], [3, 4, 5]])
        self.assertEqual(ids[0], 3)
        self.assertLess(ids.index(1), ids.index(4))

    def test_temporal_neighbors_never_cross_video(self) -> None:
        videos = {9: "A", 10: "A", 11: "A", 12: "B"}
        self.assertEqual(
            temporal_neighbor_ranking([10, 12], videos, 20), [9, 11]
        )


if __name__ == "__main__":
    unittest.main()
