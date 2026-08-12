from __future__ import annotations
import argparse
import io
import threading
import time
from pathlib import Path
import faiss
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file
from sentence_transformers import SentenceTransformer
from hybrid_search import HybridConfig, HybridSignals, reciprocal_rank_fusion
from rerank import (
    RerankConfig, SIGLIP2_BASE_MODEL, SIGLIP2_MODEL, Siglip2Reranker
)
from retrieval_enhancements import (
    expand_query, fuse_query_rankings,
)
from search import choose_device, load_metadata
from search_kis import diversify_ranked_rows, select_candidates
from submission import MAX_ANSWERS
from web_app import KeyframeStore
from assistant_service import AssistantService
from query_package import QueryPackage

MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

class MultilingualFaissEngine:
    def __init__(self, index_dir: Path, device: str = "auto",
                 hybrid_dir: Path | None = None, reranker: object | None = None,
                 query_ensemble: bool = False) -> None:
        self.device = choose_device(device)
        self.model_name = MODEL_NAME
        self.index = faiss.read_index(str(index_dir / "keyframes.faiss"))
        self.metadata_path = index_dir / "metadata.sqlite3"
        self.model = SentenceTransformer(MODEL_NAME, device=self.device)
        self._model_lock = threading.Lock()
        hybrid_dir = hybrid_dir or index_dir / "hybrid"
        self.hybrid = (
            HybridSignals(hybrid_dir, device="cpu")
            if (hybrid_dir / "hybrid_manifest.json").exists()
            else None
        )
        self.hybrid_config = HybridConfig()
        self.reranker = reranker
        self.query_ensemble = query_ensemble

    def encode(self, query: str) -> np.ndarray:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")
        with self._model_lock:
            vector = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return np.ascontiguousarray(vector, dtype=np.float32)

    def encode_many(self, queries: list[str]) -> np.ndarray:
        with self._model_lock:
            vectors = self.model.encode(
                queries, normalize_embeddings=True, convert_to_numpy=True
            )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def search(self, query: str, top_k: int, candidate_k: int = 5000,
               per_video: int = 3, min_time_gap: float = 2.0,
               quality: bool = True) -> list[dict[str, object]]:
        if top_k < 1 or top_k > MAX_ANSWERS:
            raise ValueError(f"top_k must be in [1, {MAX_ANSWERS}]")
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")
        variants = expand_query(query) if self.query_ensemble else [query]
        vectors = self.encode_many(variants)
        candidate_k = min(max(candidate_k, top_k), self.index.ntotal)
        _, ids = self.index.search(vectors, candidate_k)
        rankings = [[int(value) for value in row if value >= 0] for row in ids]
        ranked_ids, ranked_scores = fuse_query_rankings(
            rankings, limit=candidate_k
        )
        top_labels: list[str] = []
        if self.hybrid is not None:
            text_vector = self.hybrid.encode(query)
            object_ids, matched_labels = self.hybrid.object_ranking(
                text_vector, self.hybrid_config
            )
            union_ids = list(dict.fromkeys([*ranked_ids, *object_ids]))
            metadata = load_metadata(self.metadata_path, union_ids)
            video_by_id = {
                global_id: str(row["video_id"]) for global_id, row in metadata.items()
            }
            ranked_ids, ranked_scores = reciprocal_rank_fusion(
                ranked_ids, object_ids, video_by_id,
                self.hybrid.metadata_video_ranks(text_vector), self.hybrid_config
            )
            top_labels = [label for label, _ in matched_labels[:5]]
        else:
            metadata = load_metadata(self.metadata_path, ranked_ids)
        selection_size = (
            max(top_k, self.reranker.config.pool_size)
            if self.reranker and quality else top_k
        )
        selected = select_candidates(
            ranked_ids, ranked_scores, metadata, selection_size, per_video, min_time_gap,
            video_pool_limit=(max(1, selection_size // per_video)
                              if self.reranker and quality else None),
        )
        for row in selected:
            row["matched_objects"] = top_labels
        if self.reranker is not None and quality:
            selected = self.reranker.rerank(query, selected)
            selected = diversify_ranked_rows(selected, top_k, unique_prefix=20,
                                             per_video_limit=per_video)
        return selected[:top_k]

def create_app(index_dir: Path | None = None, zip_dir: Path | None = None,
               device: str = "auto", engine: object | None = None,
               keyframes: object | None = None, rerank: bool = True,
               reranker_model: str = SIGLIP2_MODEL,
               query_ensemble: bool = False,
               query_package: object | None = None) -> Flask:
    app = Flask(__name__)
    keyframe_store = keyframes or KeyframeStore(zip_dir or Path("."))
    if engine is None:
        resolved_device = choose_device(device)
        reranker = (
            Siglip2Reranker(
                keyframe_store, resolved_device, model_name=reranker_model,
                config=RerankConfig(
                    batch_size=32,
                    pool_size=120 if reranker_model == SIGLIP2_BASE_MODEL else 300
                )
            )
            if rerank else None
        )
        search_engine = MultilingualFaissEngine(
            index_dir or Path("index"), resolved_device, reranker=reranker,
            query_ensemble=query_ensemble
        )
    else:
        search_engine = engine

    assistant = AssistantService(search_engine, keyframe_store)
    package = query_package or (
        QueryPackage(Path("outputs/package_session"))
        if engine is None else QueryPackage()
    )

    @app.errorhandler(ValueError)
    def handle_value_error(error: ValueError) -> tuple[Response, int]:
        return jsonify({"error": str(error)}), 400

    @app.get("/")
    def home() -> str:
        return render_template("index_vi.html")

    @app.get("/api/health")
    def health() -> Response:
        index = getattr(search_engine, "index", None)
        return jsonify({"status": assistant.status, "model": getattr(search_engine, "model_name", MODEL_NAME),
            "device": getattr(search_engine, "device", "test"), "vectors": int(getattr(index, "ntotal", 0)),
            "videos": int(getattr(keyframe_store, "video_count", 0)),
            "index": "FAISS IndexFlatIP", "language": "Vietnamese",
            "hybrid": getattr(search_engine, "hybrid", None) is not None,
            "query_ensemble": getattr(search_engine, "query_ensemble", False),
            "reranker": (
                getattr(search_engine.reranker, "model_name", SIGLIP2_MODEL)
                if getattr(search_engine, "reranker", None) else None
            )})

    @app.post("/api/search")
    def api_search() -> Response:
        payload = request.get_json(force=True)
        started = time.perf_counter()
        query = str(payload.get("query", ""))
        use_quality = (
            bool(payload.get("quality", True))
            and getattr(search_engine, "reranker", None) is not None
        )
        with assistant.lock:
            results = search_engine.search(query, int(payload.get("top_k", 50)),
                int(payload.get("candidate_k", 5000)), int(payload.get("per_video", 3)),
                float(payload.get("min_time_gap", 2.0)), use_quality)
        return jsonify({"query": query, "count": len(results),
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "mode": "quality" if use_quality else "fast",
            "planner": (
                (("query ensemble + " if getattr(search_engine, "query_ensemble", False) else "")
                 + "clip + objects + metadata -> rrf"
                 + (" -> siglip2" if use_quality else ""))
                if getattr(search_engine, "hybrid", None) is not None
                else "multilingual-clip -> faiss -> diversify"
            ), "results": results})

    @app.post("/api/assistant")
    def api_assistant() -> Response:
        payload = request.get_json(force=True)
        return jsonify(assistant.run(
            str(payload.get("task", "kis")), str(payload.get("query", "")),
            top_k=int(payload.get("top_k", 50)),
            candidate_k=int(payload.get("candidate_k", 5000)),
            per_video=int(payload.get("per_video", 3)),
            min_time_gap=float(payload.get("min_time_gap", 2.0)),
            quality=(bool(payload.get("quality", True))
                     and getattr(search_engine, "reranker", None) is not None),
            vlm_candidates=int(payload.get("vlm_candidates", 6)),
            qa_candidates=int(payload.get("qa_candidates", 3)),
        ))
    @app.post("/api/qa/answer")
    def api_qa_answer() -> Response:
        payload = request.get_json(force=True)
        selections = payload.get("selections", [])
        if not isinstance(selections, list):
            raise ValueError("selections must be a list")
        return jsonify(assistant.answer_selected(
            str(payload.get("question", "")), selections
        ))
    @app.post("/api/package/import")
    def api_package_import() -> Response:
        uploaded = request.files.get("package")
        if uploaded is None:
            raise ValueError("Missing package ZIP")
        return jsonify({"queries": package.import_zip(uploaded.read())})

    @app.get("/api/package/status")
    def api_package_status() -> Response:
        return jsonify({"queries": package.status()})

    @app.post("/api/package/type")
    def api_package_type() -> Response:
        payload = request.get_json(force=True)
        changed = package.set_task(
            str(payload.get("query_name", "")), str(payload.get("task", ""))
        )
        return jsonify({"query": changed, "queries": package.status()})
    @app.post("/api/package/save")
    def api_package_save() -> Response:
        payload = request.get_json(force=True)
        rows = payload.get("results", [])
        if not isinstance(rows, list):
            raise ValueError("results must be a list")
        saved = package.save(str(payload.get("query_name", "")), rows)
        return jsonify({"query": saved, "queries": package.status()})

    @app.get("/api/package/export")
    def api_package_export() -> Response:
        return send_file(
            io.BytesIO(package.export_zip()), mimetype="application/zip",
            as_attachment=True, download_name="submission.zip"
        )
    @app.get("/keyframe/<video_id>/<int:keyframe_no>.jpg")
    def keyframe(video_id: str, keyframe_no: int) -> Response:
        try:
            image = keyframe_store.get_bytes(video_id, keyframe_no)
        except (KeyError, FileNotFoundError):
            return jsonify({"error": "Keyframe not found"}), 404
        return Response(io.BytesIO(image).getvalue(), mimetype="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})
    return app

def main() -> None:
    parser = argparse.ArgumentParser(description="AIC Vietnamese video search UI")
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--zip-dir", type=Path, default=Path(".."))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--query-ensemble", action="store_true")
    parser.add_argument(
        "--reranker-model", choices=[SIGLIP2_BASE_MODEL, SIGLIP2_MODEL],
        default=SIGLIP2_MODEL
    )
    args = parser.parse_args()
    create_app(
        args.index_dir, args.zip_dir, args.device, rerank=not args.no_rerank,
        reranker_model=args.reranker_model, query_ensemble=args.query_ensemble
    ).run(
        host=args.host, port=args.port, debug=False, threaded=True
    )

if __name__ == "__main__":
    main()

