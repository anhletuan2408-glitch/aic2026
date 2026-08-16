from __future__ import annotations

import gc
import threading
import time
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from qna_search import (
    QwenVLAnswerer, compose_qa_hypothesis_candidates, context_images,
    expand_qa_context_rows,
    fuse_qa_candidate_rows, rank_qa_answers,
)
from retrieval_enhancements import (
    qa_answer_hypothesis_queries,
    qa_retrieval_query,
)
from rerank import Siglip2Reranker
from search_kis import diversify_ranked_rows
from trake_search import search_trake, split_events


class AssistantService:
    """Route retrieval and answer selected Q&A frames with one GPU workload."""

    def __init__(self, engine: Any, keyframes: Any) -> None:
        self.engine = engine
        self.keyframes = keyframes
        self.lock = threading.Lock()
        self.status = "ready"

    def run(self, task: str, query: str, **options: Any) -> dict[str, Any]:
        task, query = task.strip().lower(), query.strip()
        if task not in {"kis", "qa", "trake"}:
            raise ValueError("task must be kis, qa, or trake")
        if not query:
            raise ValueError("Query must not be empty")
        with self.lock:
            self.status = "busy"
            started = time.perf_counter()
            try:
                if task == "kis":
                    results = self._retrieve(query, options, bool(options.get("quality", True)))
                elif task == "qa":
                    results = self._qa_auto(
                        query, int(options.get("qa_candidates", 3))
                    )
                else:
                    results = [{"video_id": a.video_id, "frame_ids": list(a.frame_ids)}
                               for a in search_trake(self.engine, split_events(query))]
                return {"task": task, "phase": ("answered" if task == "qa" else "retrieval"), "query": query,
                        "count": len(results),
                        "elapsed_ms": round((time.perf_counter()-started)*1000),
                        "results": results}
            finally:
                self.status = "ready"

    def _retrieve(self, query: str, options: dict[str, Any], quality: bool) -> list[dict[str, Any]]:
        return self.engine.search(query, int(options.get("top_k", 50)),
            int(options.get("candidate_k", 5000)), int(options.get("per_video", 3)),
            float(options.get("min_time_gap", 2.0)), quality)

    def _qa_auto(self, question: str, candidates: int) -> list[dict[str, Any]]:
        if candidates < 1 or candidates > 10:
            raise ValueError("qa_candidates must be in [1, 10]")
        rows = self._qa_candidate_rows(question)
        submission_rows = self._qa_submission_rows(rows, candidates)
        selections = [
            {"video_id": row["video_id"], "frame_idx": row["frame_idx"],
             "keyframe_no": row["keyframe_no"],
             **({"crop_index": row["crop_index"]} if "crop_index" in row else {}),
             "_source_index": index}
            for index, row in enumerate(rows[:candidates])
        ]
        predicted = self._answer_selected_locked(question, selections)
        answers = rank_qa_answers(
            submission_rows,
            [(int(row.pop("_source_index")), str(row["answer"])) for row in predicted],
        )
        keyframes = {
            (str(row["video_id"]), int(row["frame_idx"])): int(row["keyframe_no"])
            for row in submission_rows
        }
        return [
            {"video_id": answer.video_id, "frame_idx": answer.frame_id,
             "keyframe_no": keyframes[(answer.video_id, answer.frame_id)],
             "answer": answer.answer}
            for answer in answers
        ]
    def _qa_candidate_rows(self, question: str) -> list[dict[str, Any]]:
        scene_query = qa_retrieval_query(question)
        rows = self.engine.search(
            scene_query, 100, 10000, 5, 1.5,
            getattr(self.engine, "reranker", None) is not None, False, False,
            use_crops=True,
        )
        if scene_query.casefold() != question.casefold():
            original_rows = self.engine.search(
                question, 100, 10000, 5, 1.5, True, False, False,
                use_crops=True,
            )
            rows = fuse_qa_candidate_rows(rows, original_rows)
        hypothesis_rankings = [
            self.engine.search(
                variant, 100, 10000, 5, 1.5, True, False, False,
                use_crops=True,
            )
            for variant in qa_answer_hypothesis_queries(question)
        ]
        if hypothesis_rankings:
            rows = compose_qa_hypothesis_candidates(
                rows, hypothesis_rankings, [], hypothesis_depth=10,
            )
        return rows
    def _qa_submission_rows(
        self, rows: list[dict[str, Any]], candidates: int
    ) -> list[dict[str, Any]]:
        metadata_path = getattr(self.engine, "metadata_path", None)
        if metadata_path is None:
            return rows
        return expand_qa_context_rows(rows, metadata_path, candidates)
    def answer_selected(self, question: str,
                        selections: list[dict[str, Any]]) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        if not selections or len(selections) > 20:
            raise ValueError("Select between 1 and 20 frames")
        required = {"video_id", "frame_idx", "keyframe_no"}
        if any(not required.issubset(item) for item in selections):
            raise ValueError("Every selection needs video_id, frame_idx, and keyframe_no")
        with self.lock:
            self.status = "busy"
            started = time.perf_counter()
            try:
                results = self._answer_selected_locked(question, selections)
                return {"task": "qa", "phase": "answered", "query": question,
                        "count": len(results),
                        "elapsed_ms": round((time.perf_counter()-started)*1000),
                        "results": results}
            finally:
                self.status = "ready"

    def _answer_selected_locked(self, question: str,
                                selections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reranker = getattr(self.engine, "reranker", None)
        spec = ((reranker.model_name, reranker.config, reranker.keyframes)
                if reranker is not None else None)
        device = self.engine.device
        self.engine.model, self.engine.reranker = None, None
        del reranker
        self._clear_gpu()
        answerer = None
        try:
            answerer = QwenVLAnswerer(device)
            results = []
            for selected in selections:
                video_id = str(selected["video_id"])
                frame_idx = int(selected["frame_idx"])
                keyframe_no = int(selected["keyframe_no"])
                crop_index = (
                    int(selected["crop_index"])
                    if "crop_index" in selected else None
                )
                images = context_images(
                    self.keyframes, video_id, keyframe_no,
                    crop_index=crop_index,
                )
                if not images:
                    continue
                try:
                    answer = answerer.answer(question, images)
                finally:
                    for image in images:
                        image.close()
                result = {"video_id": video_id, "frame_idx": frame_idx,
                          "keyframe_no": keyframe_no, "answer": answer}
                if "_source_index" in selected:
                    result["_source_index"] = int(selected["_source_index"])
                results.append(result)
            if not results:
                raise RuntimeError("Qwen-VL produced no answers for selected frames")
            return results
        finally:
            if answerer is not None:
                del answerer
            self._clear_gpu()
            self.engine.model = SentenceTransformer(self.engine.model_name, device=device)
            if spec is not None:
                name, config, frames = spec
                self.engine.reranker = Siglip2Reranker(frames, device, config=config,
                                                       model_name=name)

    @staticmethod
    def _clear_gpu() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
