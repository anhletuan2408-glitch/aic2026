from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from package_submission import build_submission_zip, validate_query_csv


class SubmissionPackageTests(unittest.TestCase):
    def test_builds_required_submission_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "query-1-kis.csv").write_text(
                "L21_V001,1234\n", encoding="utf-8"
            )
            (root / "query-2-qa.csv").write_text(
                'L21_V001,1234,"Ba người, gồm hai nam"\n', encoding="utf-8"
            )
            (root / "query-3-trake.csv").write_text(
                "L21_V001,100,200,300\n", encoding="utf-8"
            )
            output = root / "round1.zip"
            build_submission_zip(root, output)
            with ZipFile(output) as archive:
                self.assertEqual(
                    archive.namelist(),
                    [
                        "submission/query-1-kis.csv",
                        "submission/query-2-qa.csv",
                        "submission/query-3-trake.csv",
                    ],
                )

    def test_header_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query-1-kis.csv"
            path.write_text("video_id,frame_idx\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid video_id"):
                validate_query_csv(path)

    def test_trake_requires_equal_event_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query-1-trake.csv"
            path.write_text(
                "L21_V001,100,200\nL21_V002,100,200,300\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "equal event counts"):
                validate_query_csv(path)


if __name__ == "__main__":
    unittest.main()
