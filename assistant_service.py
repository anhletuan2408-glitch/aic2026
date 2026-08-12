from __future__ import annotations

import gc
import threading
import time
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from qna_search import QwenVLAnswerer, answer_rows
from rerank import Siglip2Reranker
from trake_search import search_trake, split_events


class AssistantService:
    """Route preliminary tasks while allowing one GPU workload at a time."""

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
                    results = self._kis(query, options)
                elif task == "trake":
                    results = [{"video_id": a.video_id, "frame_ids": list(a.frame_ids)}
                               for a in search_trake(self.engine, split_events(query))]
                else:
                    results = self._qa(query, int(options.get("vlm_candidates", 6)))
                return {"task": task, "query": query, "count": len(results),
                        "elapsed_ms": round((time.perf_counter()-started)*1000),
                        "results": results}
            finally:
                self.status = "ready"

    def _kis(self, query: str, options: dict[str, Any]) -> list[dict[str, Any]]:
        return self.engine.search(query, int(options.get("top_k", 50)),
            int(options.get("candidate_k", 5000)), int(options.get("per_video", 3)),
            float(options.get("min_time_gap", 2.0)), bool(options.get("quality", True)))

    def _qa(self, question: str, candidates: int) -> list[dict[str, Any]]:
        if candidates < 1 or candidates > 20:
            raise ValueError("vlm_candidates must be in [1, 20]")
        rows = self.engine.search(question, 100, 5000, 3, 2.0, False)
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
            answers = answer_rows(rows, self.keyframes, answerer, question, candidates)
            keyframes = {(str(r["video_id"]), int(r["frame_idx"])): int(r["keyframe_no"])
                         for r in rows}
            return [{"video_id": a.video_id, "frame_idx": a.frame_id,
                     "keyframe_no": keyframes.get((a.video_id, a.frame_id)),
                     "answer": a.answer} for a in answers]
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