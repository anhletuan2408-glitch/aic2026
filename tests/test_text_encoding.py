from __future__ import annotations

import unittest

from text_encoding import repair_utf8_mojibake


class TextEncodingTests(unittest.TestCase):
    def test_repairs_vietnamese_utf8_decoded_as_cp1252(self) -> None:
        self.assertEqual(repair_utf8_mojibake("NÄƒm ngÆ°á»i"), "Năm người")
        self.assertEqual(
            repair_utf8_mojibake("MÃ u Ä‘á», ráº¥t Ä‘áº¹p"),
            "Màu đỏ, rất đẹp",
        )

    def test_preserves_valid_unicode_and_unrepairable_text(self) -> None:
        self.assertEqual(repair_utf8_mojibake("Năm người"), "Năm người")
        self.assertEqual(repair_utf8_mojibake("Äpfel"), "Äpfel")


if __name__ == "__main__":
    unittest.main()
