import io
import json
import threading
from types import SimpleNamespace
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
import faiss
import numpy as np
from web_app_vi import (
    MODEL_NAME, MultilingualFaissEngine, create_app, load_siglip2_index,
)
from ground_truth_store import GroundTruthStore

class FakeEngine:
    model_name=MODEL_NAME; device="cpu"; index=SimpleNamespace(ntotal=177321); reranker=object()
    def __init__(self): self.queries=[]
    def search(self,query,top_k,candidate_k,per_video,min_time_gap,quality=True):
        if not query.strip(): raise ValueError("Query must not be empty")
        self.queries.append((query,quality))
        return [{"video_id":"L21_V001","keyframe_no":12,"frame_idx":345,"score":0.42,"pts_time":11.5}]

class FakeFrames:
    video_count=873
    def get_bytes(self,video_id,keyframe_no):
        if video_id!="L21_V001": raise KeyError(video_id)
        return b"jpeg"

class WebAppVietnameseTests(unittest.TestCase):
    def setUp(self):
        self.engine=FakeEngine()
        self.temp=tempfile.TemporaryDirectory()
        self.gt=GroundTruthStore(Path(self.temp.name)/"local.jsonl")
        app=create_app(engine=self.engine,keyframes=FakeFrames(),ground_truth=self.gt)
        app.testing=True
        self.client=app.test_client()

    def tearDown(self):
        self.temp.cleanup()

    def test_quality_search_fuses_global_siglip_but_fast_search_skips_it(self):
        class TextModel:
            def encode(self, values, **_kwargs):
                return np.ones((len(values), 2), dtype=np.float32)

        class Index:
            ntotal = 3

            def __init__(self, ids):
                self.ids = np.asarray([ids], dtype=np.int64)

            def search(self, _vectors, _count):
                scores = np.asarray([[.9, .8, .7]], dtype=np.float32)
                return scores, self.ids

        class Reranker:
            config = SimpleNamespace(pool_size=3)

            def __init__(self):
                self.queries = []
                self.rerank_calls = 0

            def encode_text(self, query):
                self.queries.append(query)
                return np.asarray([[1.0, 0.0]], dtype=np.float32)

            def rerank(self, _query, rows):
                self.rerank_calls += 1
                return rows

        engine = MultilingualFaissEngine.__new__(MultilingualFaissEngine)
        engine.model = TextModel()
        engine._model_lock = threading.Lock()
        engine.index = Index([0, 1, 2])
        engine.siglip2_index = Index([1, 2, 0])
        engine.metadata_path = Path("unused")
        engine.hybrid = None
        engine.ocr = None
        engine.query_ensemble = False
        engine.reranker = Reranker()

        def select(ids, scores, _metadata, size, *_args, **_kwargs):
            return [{
                "video_id": f"L21_V00{global_id + 1}",
                "frame_idx": global_id,
                "keyframe_no": global_id,
                "pts_time": float(global_id),
                "score": float(score),
                "_global_id": global_id,
            } for global_id, score in zip(ids[:size], scores[:size])]

        with patch("web_app_vi.load_metadata", return_value={}), \
             patch("web_app_vi.select_candidates", side_effect=select), \
             patch("web_app_vi.diversify_ranked_rows", side_effect=lambda rows, *_a, **_k: rows), \
             patch("web_app_vi.protect_signal_rows", side_effect=lambda rows: rows):
            quality = engine.search("motorcycle", 2, candidate_k=3, quality=True)
            fast = engine.search("motorcycle", 2, candidate_k=3, quality=False)
            engine.siglip2_index = None
            engine.search("motorcycle", 2, candidate_k=3, quality=True)

        self.assertEqual(quality[0]["frame_idx"], 1)
        self.assertEqual(fast[0]["frame_idx"], 0)
        self.assertEqual(engine.reranker.queries, ["motorcycle"])
        self.assertEqual(engine.reranker.rerank_calls, 1)

    def test_global_siglip_index_requires_complete_matching_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            siglip = root / "siglip2"
            siglip.mkdir()
            index = faiss.IndexFlatIP(4)
            index.add(np.eye(4, dtype=np.float32)[:2])
            faiss.write_index(index, str(siglip / "keyframes.faiss"))
            manifest = {
                "model": "model/test", "frames": 2, "completed": 2,
                "index_frames": 2, "dimension": 4,
            }
            (siglip / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            loaded = load_siglip2_index(root, 2, "model/test")
            self.assertEqual(loaded.ntotal, 2)
            manifest["completed"] = 1
            (siglip / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "partial"):
                load_siglip2_index(root, 2, "model/test")

    def test_in_progress_siglip_manifest_does_not_break_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            siglip = root / "siglip2"
            siglip.mkdir()
            (siglip / "manifest.json").write_text(json.dumps({
                "model": "model/test", "frames": 2, "completed": 0,
                "index_frames": 0, "dimension": 4,
            }), encoding="utf-8")
            self.assertIsNone(load_siglip2_index(root, 2, "model/test"))
            index = faiss.IndexFlatIP(4)
            index.add(np.eye(4, dtype=np.float32)[:2])
            faiss.write_index(index, str(siglip / "keyframes.faiss"))
            self.assertIsNone(load_siglip2_index(root, 2, "model/test"))

    def test_ground_truth_api_upserts_record(self):
        payload={"query_id":"kis-1","task":"kis","query":"red car",
                 "video_id":"L21_V001","start":300,"end":400}
        response=self.client.post("/api/ground-truth",json=payload)
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.get_json()["count"],1)
        data=self.client.get("/api/ground-truth").get_json()
        self.assertEqual(data["records"][0]["query_id"],"kis-1")
    def test_health(self):
        data=self.client.get("/api/health").get_json()
        self.assertEqual((data["model"],data["vectors"],data["videos"],data["language"]),
                         (MODEL_NAME,177321,873,"Vietnamese"))

    def test_short_and_long_queries_are_forwarded_unchanged(self):
        queries=["xe máy","Một người phụ nữ mặc áo đỏ đang cầm micro phát biểu trên sân khấu"]
        for query in queries:
            response=self.client.post("/api/search",json={"query":query,"top_k":50})
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.get_json()["query"],query)
        self.assertEqual(self.engine.queries,[(query,True) for query in queries])
    def test_fast_mode_is_forwarded(self):
        response=self.client.post("/api/search",json={"query":"xe máy","quality":False})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.get_json()["mode"],"fast")
        self.assertEqual(self.engine.queries[-1],("xe máy",False))

    def test_assistant_kis_route(self):
        response=self.client.post("/api/assistant",json={"task":"kis","query":"xe máy","top_k":1,"quality":False})
        self.assertEqual(response.status_code,200)
        self.assertEqual((response.get_json()["task"],response.get_json()["count"]),("kis",1))
    def test_qa_answer_route_requires_selection(self):
        response=self.client.post("/api/qa/answer",json={"question":"Màu gì?","selections":[]})
        self.assertEqual(response.status_code,400)
        self.assertIn("Select between",response.get_json()["error"])
    def test_import_save_and_export_btc_package(self):
        source=io.BytesIO()
        with ZipFile(source,"w") as archive:
            archive.writestr("query-1-kis.txt","một người mở laptop")
        response=self.client.post("/api/package/import",data={
            "package":(io.BytesIO(source.getvalue()),"round1.zip")
        },content_type="multipart/form-data")
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.get_json()["queries"][0]["task"],"kis")
        response=self.client.post("/api/package/save",json={
            "query_name":"query-1-kis.txt",
            "results":[{"video_id":"L21_V001","frame_idx":345}],
        })
        self.assertEqual(response.status_code,200)
        response=self.client.get("/api/package/export")
        self.assertEqual(response.status_code,200)
        with ZipFile(io.BytesIO(response.data)) as archive:
            self.assertEqual(archive.namelist(),["submission/query-1-kis.csv"])
            self.assertEqual(archive.read(archive.namelist()[0]),b"L21_V001,345\r\n")
    def test_package_auto_run_finishes_before_human_review(self):
        source = io.BytesIO()
        with ZipFile(source, "w") as archive:
            archive.writestr("query-1-kis.txt", "red car")
        self.client.post("/api/package/import", data={
            "package": (io.BytesIO(source.getvalue()), "round1.zip")
        }, content_type="multipart/form-data")
        started = self.client.post("/api/package/auto-run", json={})
        self.assertEqual(started.status_code, 202)
        for _ in range(100):
            data = self.client.get("/api/package/auto-status").get_json()
            if not data["auto"]["running"]:
                break
            threading.Event().wait(0.01)
        self.assertFalse(data["auto"]["running"])
        self.assertTrue(data["queries"][0]["completed"])
        self.assertIn(
            data["queries"][0]["review_status"],
            {"needs_review", "auto_accepted"},
        )

    def test_empty_query_and_keyframe(self):
        self.assertEqual(self.client.post("/api/search",json={"query":" "}).status_code,400)
        response=self.client.get("/keyframe/L21_V001/12.jpg")
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.data,b"jpeg")

