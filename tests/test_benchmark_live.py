from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from benchmark_live import call_api, result_rows, write_prediction


class BenchmarkLiveTests(unittest.TestCase):
    @patch("benchmark_live.wait_for_health")
    @patch("benchmark_live.urllib.request.urlopen")
    def test_api_retries_after_connection_reset(self, urlopen, wait_health) -> None:
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b'{"count": 1, "results": []}'
        urlopen.side_effect = [ConnectionResetError("reset"), response]
        result = call_api(
            "http://127.0.0.1:7860",
            {"query_id":"qa-1","task":"qa","query":"question"},
            qa_candidates=5,
        )
        self.assertEqual(result["count"], 1)
        wait_health.assert_called_once()

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