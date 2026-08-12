from __future__ import annotations

import unittest

from retrieval_enhancements import (
    expand_query, fuse_query_rankings, temporal_neighbor_ranking,
)


class RetrievalEnhancementTests(unittest.TestCase):
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
