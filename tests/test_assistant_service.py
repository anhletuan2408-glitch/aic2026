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

    def test_qa_retrieves_scene_and_original_question(self):
        self.engine.search = Mock(return_value=[
            {"video_id":"L21_V001","frame_idx":345,"keyframe_no":12,"score":.5}
        ])
        with patch.object(
            self.service, "_answer_selected_locked",
            return_value=[{"_source_index":0,"answer":"1"}],
        ):
            self.service.run(
                "qa",
                "C\u00f3 bao nhi\u00eau ng\u01b0\u1eddi \u0111\u1ee9ng tr\u01b0\u1edbc b\u1ea3ng tr\u1eafng?",
                qa_candidates=1,
            )
        self.assertEqual(self.engine.search.call_count, 8)
        self.assertNotEqual(
            self.engine.search.call_args_list[0].args[0],
            self.engine.search.call_args_list[1].args[0],
        )
        for call in self.engine.search.call_args_list:
            self.assertFalse(call.args[-2])
            self.assertFalse(call.args[-1])

    def test_answer_requires_selected_frames(self):
        with self.assertRaises(ValueError):
            self.service.answer_selected("Question?", [])
