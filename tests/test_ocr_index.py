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


if __name__ == "__main__":
    unittest.main()