import io
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from query_package import QueryPackage


def package_zip(files: dict[str, str]) -> bytes:
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return stream.getvalue()


class QueryPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.package = QueryPackage()
        self.queries = {
            "query-1-kis.txt": "một người mở laptop",
            "query-2-qa.txt": "người phụ nữ đi phương tiện gì?",
            "query-3-trake.txt": "đứng lên; phát biểu; ngồi xuống",
        }

    def test_import_preserves_names_and_detects_tasks(self) -> None:
        status = self.package.import_zip(package_zip(self.queries))
        self.assertEqual([item["task"] for item in status], ["kis", "qa", "trake"])
        self.assertEqual(status[1]["output_name"], "query-2-qa.csv")

    def test_rejects_nested_or_unknown_files(self) -> None:
        for name in ("folder/query-1-kis.txt", "query-1.exe"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.package.import_zip(package_zip({name: "query"}))

    def test_arbitrary_txt_name_can_choose_type(self) -> None:
        status = self.package.import_zip(package_zip({"cau-hoi-01.txt": "Màu gì?"}))
        self.assertEqual((status[0]["task"], status[0]["task_suggested"]), ("kis", False))
        changed = self.package.set_task("cau-hoi-01.txt", "qa")
        self.assertEqual((changed["task"], changed["output_name"]), ("qa", "cau-hoi-01.csv"))
        self.package.save("cau-hoi-01.txt", [
            {"video_id": "L21_V001", "frame_idx": 345, "answer": "đỏ"}
        ])

    def test_changing_type_resets_saved_result(self) -> None:
        self.package.import_zip(package_zip({"query-1-kis.txt": "mở laptop"}))
        self.package.save("query-1-kis.txt", [
            {"video_id": "L21_V001", "frame_idx": 345}
        ])
        changed = self.package.set_task("query-1-kis.txt", "qa")
        self.assertFalse(changed["completed"])
        with self.assertRaisesRegex(ValueError, "Incomplete queries"):
            self.package.export_zip()
    def test_save_all_and_export_submission_root(self) -> None:
        self.package.import_zip(package_zip(self.queries))
        self.package.save("query-1-kis.txt", [
            {"video_id": "L21_V001", "frame_idx": 345}
        ])
        self.package.save("query-2-qa.txt", [
            {"video_id": "L22_V031", "frame_idx": 28372, "answer": "xe máy"}
        ])
        self.package.save("query-3-trake.txt", [
            {"video_id": "L22_V003", "frame_ids": [404, 2476, 26202]}
        ])
        with ZipFile(io.BytesIO(self.package.export_zip())) as archive:
            self.assertEqual(archive.namelist(), [
                "submission/query-1-kis.csv",
                "submission/query-2-qa.csv",
                "submission/query-3-trake.csv",
            ])
            self.assertEqual(
                archive.read("submission/query-2-qa.csv").decode(),
                "L22_V031,28372,xe máy\r\n",
            )

    def test_qa_csv_preserves_whitespace_and_escapes_special_characters(self) -> None:
        self.package.import_zip(package_zip({"query-1-qa.txt": "Người đó nói gì?"}))
        answer = ' Anh ấy nói "Xin chào", rồi rời đi '
        self.package.save("query-1-qa.txt", [
            {"video_id": "L21_V001", "frame_idx": 1234, "answer": answer}
        ])
        with ZipFile(io.BytesIO(self.package.export_zip())) as archive:
            content = archive.read("submission/query-1-qa.csv").decode("utf-8")
        self.assertEqual(
            content,
            'L21_V001,1234," Anh ấy nói ""Xin chào"", rồi rời đi "\r\n',
        )
    def test_saved_progress_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            package = QueryPackage(workspace)
            package.import_zip(package_zip({
                "query-1-kis.txt": "một người mở laptop"
            }))
            package.save("query-1-kis.txt", [
                {"video_id": "L21_V001", "frame_idx": 345}
            ])
            restored = QueryPackage(workspace)
            self.assertTrue(restored.status()[0]["completed"])
            with ZipFile(io.BytesIO(restored.export_zip())) as archive:
                self.assertEqual(archive.namelist(), [
                    "submission/query-1-kis.csv"
                ])
    def test_export_rejects_incomplete_package(self) -> None:
        self.package.import_zip(package_zip(self.queries))
        with self.assertRaisesRegex(ValueError, "Incomplete queries"):
            self.package.export_zip()

    def test_auto_result_enters_review_queue_and_human_save_clears_it(self) -> None:
        self.package.import_zip(package_zip({"query-1-kis.txt": "red car"}))
        rows = [
            {"video_id": "L21_V001", "frame_idx": 100, "score": 0.10},
            {"video_id": "L22_V001", "frame_idx": 200, "score": 0.099},
        ]
        saved = self.package.save("query-1-kis.txt", rows)
        self.assertEqual(saved["review_status"], "needs_review")
        self.assertEqual(len(self.package.review_queue()), 1)
        reviewed = self.package.save(
            "query-1-kis.txt", rows, human_reviewed=True
        )
        self.assertEqual(reviewed["review_status"], "reviewed")
        self.assertEqual(self.package.review_queue(), [])



if __name__ == "__main__":
    unittest.main()
