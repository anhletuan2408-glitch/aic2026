from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from diagnose_signal_ranks import first_target_rank, target_global_ids


class SignalRankDiagnosticsTests(unittest.TestCase):
    def test_first_target_rank_reports_coverage(self) -> None:
        evidence = first_target_rank([9, 4, 8, 3, 4], {3, 4})
        self.assertEqual(evidence.rank, 2)
        self.assertEqual(evidence.searched, 5)
        self.assertEqual(evidence.target_hits, 3)

    def test_missing_target_rank_is_none(self) -> None:
        evidence = first_target_rank([1, 2, 3], {8})
        self.assertIsNone(evidence.rank)
        self.assertEqual(evidence.target_hits, 0)

    def test_target_ids_use_video_and_frame_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.sqlite3"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute(
                    "CREATE TABLE keyframes("
                    "global_id INTEGER,video_id TEXT,frame_idx INTEGER)"
                )
                connection.executemany(
                    "INSERT INTO keyframes VALUES(?,?,?)",
                    [(0, "L01_V001", 10), (1, "L01_V001", 20),
                     (2, "L01_V002", 20), (3, "L01_V001", 30)],
                )
                connection.commit()
            ids = target_global_ids(path, "L01_V001", 15, 25)
        self.assertEqual(ids, {1})


if __name__ == "__main__":
    unittest.main()
