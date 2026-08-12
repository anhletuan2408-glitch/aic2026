from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from assistant_service import AssistantService


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

    def test_kis_retrieval_and_invalid_task(self):
        kis = self.service.run("kis", "xe may", top_k=1, quality=False)
        self.assertEqual((kis["task"], kis["count"]), ("kis", 1))
        with self.assertRaises(ValueError):
            self.service.run("auto", "query")

    def test_status_recovers_after_failure(self):
        with patch.object(self.service, "_retrieve", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self.service.run("kis", "query")
        self.assertEqual(self.service.status, "ready")

    @patch("assistant_service.SentenceTransformer", return_value="restored")
    @patch("assistant_service.context_images")
    @patch("assistant_service.QwenVLAnswerer")
    def test_qwen_answers_only_selected_frame(self, qwen, images, _clip):
        image = Mock()
        images.return_value = [image]
        qwen.return_value = SimpleNamespace(answer=lambda question, frames: "xe máy")
        selected = [{"video_id":"L21_V001","frame_idx":345,"keyframe_no":12}]
        data = self.service.answer_selected("Phương tiện gì?", selected)
        self.assertEqual(data["phase"], "answered")
        self.assertEqual(data["results"], [{"video_id":"L21_V001","frame_idx":345,
                                             "keyframe_no":12,"answer":"xe máy"}])
        image.close.assert_called_once()
        self.assertEqual(self.engine.model, "restored")

    @patch("assistant_service.SentenceTransformer", return_value="restored")
    @patch("assistant_service.context_images")
    @patch("assistant_service.QwenVLAnswerer")
    def test_qa_runs_retrieval_and_answer_in_one_request(self, qwen, images, _clip):
        image = Mock()
        images.return_value = [image]
        qwen.return_value = SimpleNamespace(answer=lambda question, frames: "xe may")
        data = self.service.run("qa", "Phuong tien gi?", qa_candidates=1)
        self.assertEqual(data["phase"], "answered")
        self.assertEqual(data["results"], [{"video_id":"L21_V001","frame_idx":345,
                                             "keyframe_no":12,"answer":"xe may"}])
        image.close.assert_called_once()
        self.assertEqual(self.engine.model, "restored")

    def test_answer_requires_selected_frames(self):
        with self.assertRaises(ValueError):
            self.service.answer_selected("Question?", [])
