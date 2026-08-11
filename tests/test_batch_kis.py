from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from batch_kis import load_queries


class BatchQueryLoadingTests(unittest.TestCase):
    def test_loads_jsonl_and_honors_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            rows = [
                {"query_id": "q1", "query": "first", "variants": ["one"]},
                {"query_id": "q2", "query": "second", "variants": []},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            loaded = load_queries(path, limit=1)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["query_id"], "q1")
        self.assertEqual(loaded[0]["texts"], ["first", "one"])

    def test_duplicate_query_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            path.write_text(
                '{"query_id":"q1","query":"first"}\n'
                '{"query_id":"q1","query":"second"}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_queries(path)


if __name__ == "__main__":
    unittest.main()
