from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from web_app import KeyframeStore, create_app


JPEG = b"\xff\xd8\xff\xd9"


class FakeEngine:
    index = SimpleNamespace(ntotal=177321)

    def search(
        self,
        query: str,
        variants: list[str],
        top_k: int,
        candidate_k: int,
        per_video: int,
        min_time_gap: float,
    ) -> list[dict[str, object]]:
        if not query.strip():
            raise ValueError("query required")
        return [
            {
                "rank": 1,
                "retrieval_rank": 1,
                "video_id": "L21_V001",
                "keyframe_no": 1,
                "frame_idx": 0,
                "pts_time": 0.0,
                "score": 0.9,
                "title": "Example",
            }
        ][:top_k]


class FakeStore:
    video_count = 1

    def get_bytes(self, video_id: str, keyframe_no: int) -> bytes:
        if video_id != "L21_V001" or keyframe_no != 1:
            raise KeyError(video_id)
        return JPEG


class WebAppTests(unittest.TestCase):
    def setUp(self) -> None:
        app = create_app(engine=FakeEngine(), keyframes=FakeStore())
        app.testing = True
        self.client = app.test_client()

    def test_health_and_home(self) -> None:
        self.assertEqual(self.client.get("/").status_code, 200)
        health = self.client.get("/api/health").get_json()
        self.assertEqual(health, {"status": "ok", "vectors": 177321, "videos": 1})

    def test_search_and_keyframe(self) -> None:
        response = self.client.post(
            "/api/search",
            json={"query": "a speaker", "variants": [], "top_k": 20},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["results"][0]["frame_idx"], 0)
        image = self.client.get("/keyframe/L21_V001/1.jpg")
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.data, JPEG)

    def test_keyframe_store_reads_zip_without_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            zip_path = Path(directory) / "Keyframes_Test.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("keyframes/L21_V001/001.jpg", JPEG)
            store = KeyframeStore(Path(directory))
            try:
                self.assertEqual(store.video_count, 1)
                self.assertEqual(store.get_bytes("L21_V001", 1), JPEG)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
