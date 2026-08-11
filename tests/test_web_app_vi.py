from types import SimpleNamespace
import unittest
from web_app_vi import MODEL_NAME, create_app

class FakeEngine:
    model_name=MODEL_NAME; device="cpu"; index=SimpleNamespace(ntotal=177321)
    def __init__(self): self.queries=[]
    def search(self,query,top_k,candidate_k,per_video,min_time_gap):
        if not query.strip(): raise ValueError("Query must not be empty")
        self.queries.append(query)
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
        self.assertEqual(self.engine.queries,queries)

    def test_empty_query_and_keyframe(self):
        self.assertEqual(self.client.post("/api/search",json={"query":" "}).status_code,400)
        response=self.client.get("/keyframe/L21_V001/12.jpg")
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.data,b"jpeg")

