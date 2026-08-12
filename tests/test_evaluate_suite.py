from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluate_suite import evaluate_suite, load_ground_truth


class EvaluateSuiteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.truth = self.root / "truth.jsonl"
        records = [
            {"query_id":"kis-1","task":"kis","query":"moment","video_id":"L21_V001","start":100,"end":200},
            {"query_id":"qa-1","task":"qa","query":"color?","video_id":"L21_V002","start":300,"end":400,"answers":["red"]},
            {"query_id":"trake-1","task":"trake","query":"a;b","video_id":"L21_V003","moments":[[10,20],[30,40]]},
        ]
        self.truth.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
        self.predictions = self.root / "predictions"
        self.predictions.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, name: str, rows: list[list[object]]) -> None:
        with (self.predictions / name).open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)

    def test_scores_all_three_tasks(self) -> None:
        self.write("kis-1.csv", [["L21_V001", 150]])
        self.write("qa-1.csv", [["L21_V002", 350, "red"]])
        self.write("trake-1.csv", [["L21_V003", 15, 35]])
        report = evaluate_suite(self.truth, self.predictions)
        self.assertEqual(report["queries"], 3)
        self.assertEqual(report["missing_predictions"], 0)
        self.assertEqual(report["overall"]["final_score"], 1.0)
        self.assertEqual(set(report["tasks"]), {"kis", "qa", "trake"})

    def test_missing_prediction_scores_zero(self) -> None:
        report = evaluate_suite(self.truth, self.predictions)
        self.assertEqual(report["missing_predictions"], 3)
        self.assertEqual(report["overall"]["final_score"], 0.0)

    def test_duplicate_query_is_rejected(self) -> None:
        line = {"query_id":"x","task":"kis","query":"q","video_id":"L21_V001","start":1,"end":2}
        self.truth.write_text(json.dumps(line)+"\n"+json.dumps(line), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_ground_truth(self.truth)


if __name__ == "__main__":
    unittest.main()