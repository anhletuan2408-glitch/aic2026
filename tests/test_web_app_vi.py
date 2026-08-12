import io
from types import SimpleNamespace
import unittest
from zipfile import ZipFile
from web_app_vi import MODEL_NAME, create_app

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
        app=create_app(engine=self.engine,keyframes=FakeFrames())
        app.testing=True
        self.client=app.test_client()

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
    def test_empty_query_and_keyframe(self):
        self.assertEqual(self.client.post("/api/search",json={"query":" "}).status_code,400)
        response=self.client.get("/keyframe/L21_V001/12.jpg")
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.data,b"jpeg")

