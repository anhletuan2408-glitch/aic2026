from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from assistant_service import AssistantService
from submission import QAAnswer


class FakeEngine:
    model_name = "clip"
    device = "cpu"
    reranker = None
    model = object()
    def search(self, *args):
        return [{"video_id":"L21_V001","frame_idx":345,"keyframe_no":12,"score":.5}]


class AssistantServiceTests(TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.service = AssistantService(self.engine, object())

    def test_kis_and_invalid_task(self):
        data = self.service.run("kis", "xe máy", top_k=1, quality=False)
        self.assertEqual((data["task"], data["count"]), ("kis", 1))
        with self.assertRaises(ValueError):
            self.service.run("auto", "query")

    def test_status_recovers_after_failure(self):
        with patch.object(self.service, "_kis", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self.service.run("kis", "query")
        self.assertEqual(self.service.status, "ready")
    @patch("assistant_service.SentenceTransformer", return_value="restored")
    @patch("assistant_service.answer_rows", return_value=[QAAnswer("L21_V001",345,"xe máy")])
    @patch("assistant_service.QwenVLAnswerer", return_value=SimpleNamespace())
    def test_qa_swaps_model_and_restores_retrieval(self, _qwen, _answers, _clip):
        data = self.service.run("qa", "Phương tiện gì?", vlm_candidates=3)
        self.assertEqual(data["results"][0]["answer"], "xe máy")
        self.assertEqual(data["results"][0]["keyframe_no"], 12)
        self.assertEqual(self.engine.model, "restored")