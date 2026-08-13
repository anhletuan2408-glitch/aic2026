from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ocr_index import OCRSignals, connect_ocr, fts_query, insert_batch


class OCRIndexTests(unittest.TestCase):
    def test_fts_search_is_unicode_and_ranked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ocr.sqlite3"
            connection = connect_ocr(path)
            insert_batch(connection, [
                (10,"L21_V001",1,100,"M\u00e0u \u0111\u1ecf r\u1ea5t \u0111\u1eb9p",.95),
                (20,"L21_V002",1,200,"M\u00e0u xanh",.90),
                (30,"L21_V003",1,300,"Kh\u00f4ng c\u00f3 ch\u1eef",.80),
            ])
            connection.close()
            ranked = OCRSignals(path).ranking("m\u00e0u \u0111\u1ecf")
            self.assertEqual(ranked[0], 10)
            self.assertEqual(OCRSignals(path).count(), 3)

    def test_empty_or_punctuation_query_does_not_match_everything(self) -> None:
        self.assertEqual(fts_query("?! ."), "")


    def test_long_visual_query_uses_multiple_ocr_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ocr.sqlite3"
            connection = connect_ocr(path)
            insert_batch(connection, [
                (10, "L21_V001", 1, 100,
                 "Tuy\u1ebfn B\u1ebfn L\u1ee9c Trung L\u01b0\u01a1ng m\u1edf r\u1ed9ng 10 l\u00e0n", .95),
                (20, "L21_V002", 1, 200, "C\u00e1nh \u0111\u1ed3ng xanh", .90),
                (30, "L21_V003", 1, 300, "M\u1ed9t tuy\u1ebfn \u0111\u01b0\u1eddng kh\u00e1c", .90),
            ])
            connection.close()
            ranked = OCRSignals(path).ranking(
                "\u1ea3nh ch\u1ee5p t\u1eeb tr\u00ean cao c\u00e1nh \u0111\u1ed3ng v\u00e0 "
                "tuy\u1ebfn B\u1ebfn L\u1ee9c Trung L\u01b0\u01a1ng"
            )
            self.assertEqual(ranked[0], 10)
            self.assertNotIn(30, ranked)

if __name__ == "__main__":
    unittest.main()