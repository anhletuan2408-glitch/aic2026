from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ground_truth_store import GroundTruthStore


class GroundTruthStoreTests(unittest.TestCase):
    def test_upserts_all_task_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = GroundTruthStore(Path(directory) / "local.jsonl")
            store.upsert({"query_id":"kis-1","task":"kis","query":"q","video_id":"L21_V001","start":1,"end":2})
            store.upsert({"query_id":"qa-1","task":"qa","query":"q?","video_id":"L21_V002","start":3,"end":4,"answers":["red"]})
            store.upsert({"query_id":"trake-1","task":"trake","query":"a;b","video_id":"L21_V003","moments":[[1,2],[3,4]]})
            store.upsert({"query_id":"qa-1","task":"qa","query":"q?","video_id":"L21_V002","start":3,"end":5,"answers":["red","đỏ"]})
            records = store.records()
            self.assertEqual(len(records), 3)
            self.assertEqual(records[1]["end"], 5)
            self.assertEqual(records[2]["moments"], [[1,2],[3,4]])

    def test_invalid_records_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = GroundTruthStore(Path(directory) / "local.jsonl")
            with self.assertRaises(ValueError):
                store.upsert({"query_id":"bad id","task":"kis","query":"q","video_id":"oops","start":2,"end":1})


if __name__ == "__main__":
    unittest.main()