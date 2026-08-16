from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from validation.append_query import append_record, make_record


class ValidationAppendTests(unittest.TestCase):
    def test_make_and_append_record(self) -> None:
        record = make_record(
            "q1", "  xe   máy  ", "L21_V001", 10, 20, "round-1"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            append_record(path, record)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["query"], "xe máy")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                append_record(path, record)

    def test_invalid_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame range"):
            make_record("q1", "query", "L21_V001", 20, 10, "round-1")


if __name__ == "__main__":
    unittest.main()
