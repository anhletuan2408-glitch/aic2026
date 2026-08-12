from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from benchmark_live import result_rows, write_prediction


class BenchmarkLiveTests(unittest.TestCase):
    def test_result_rows_match_competition_schemas(self) -> None:
        self.assertEqual(result_rows("kis", [{"video_id":"L21_V001","frame_idx":10}]),
                         [["L21_V001",10]])
        self.assertEqual(result_rows("qa", [{"video_id":"L21_V001","frame_idx":10,"answer":"red"}]),
                         [["L21_V001",10,"red"]])
        self.assertEqual(result_rows("trake", [{"video_id":"L21_V001","frame_ids":[10,20]}]),
                         [["L21_V001",10,20]])

    def test_atomic_writer_has_no_header_and_caps_at_100(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.csv"
            write_prediction(path, [["L21_V001", value] for value in range(120)])
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(len(rows), 100)
            self.assertEqual(rows[0], ["L21_V001", "0"])


if __name__ == "__main__":
    unittest.main()