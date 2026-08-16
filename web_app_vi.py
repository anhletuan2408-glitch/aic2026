from __future__ import annotations
import argparse
import io
import json
import threading
import time
from pathlib import Path
import faiss
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request, send_file
from sentence_transformers import SentenceTransformer
from hybrid_search import HybridConfig, HybridSignals, reciprocal_rank_fusion
from multicrop import load_multicrop_index, search_multicrop
from ocr_index import OCRSignals, has_ocr_intent
from rerank import (
    RerankConfig, SIGLIP2_BASE_MODEL, SIGLIP2_MODEL, Siglip2Reranker
)
from retrieval_enhancements import (
    expand_query, fuse_query_rankings, normalize_visual_query,
)
from search import choose_device, load_metadata
from search_kis import (
    diversify_ranked_rows, protect_signal_ids, protect_signal_rows,
    select_candidates,
)
from submission import MAX_ANSWERS
from web_app import KeyframeStore
from assistant_service import AssistantService
from query_package import QueryPackage
from ground_truth_store import GroundTruthStore

MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
SIGLIP2_GLOBAL_WEIGHT = 1.5


def load_siglip2_index(index_dir: Path, expected_frames: int,
                        model_name: str) -> object | None:
    directory = index_dir / "siglip2"
    index_path = directory / "keyframes.faiss"
    manifest_path = directory / "manifest.json"
    if not index_path.exists() and not manifest_path.exists():
        return None
    if not manifest_path.exists():
        raise ValueError("SigLIP2 keyframes.faiss is missing manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("model")) != model_name:
        raise ValueError("SigLIP2 index model does not match the configured reranker")
    frames = int(manifest.get("frames", -1))
    completed = int(manifest.get("completed", -1))
    indexed = int(manifest.get("index_frames", -1))
    if completed < frames and indexed == 0:
        return None
    if not index_path.exists():
        raise ValueError("Completed SigLIP2 manifest is missing keyframes.faiss")
    if frames != expected_frames or completed != frames or indexed != frames:
        raise ValueError("SigLIP2 index is partial or does not match the base index")
    index = faiss.read_index(str(index_path))
    if index.ntotal != expected_frames or index.d != int(manifest["dimension"]):
        raise ValueError("SigLIP2 FAISS dimensions/count do not match its manifest")
    return index


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
        ocr_path = index_dir / "ocr.sqlite3"
        self.ocr = OCRSignals(ocr_path) if ocr_path.exists() else None
        self.reranker = reranker
        self.siglip2_index = (
            load_siglip2_index(index_dir, self.index.ntotal, reranker.model_name)
            if reranker is not None else None
        )
        crop_state = (
            load_multicrop_index(
                index_dir / "siglip2-crops", self.index.ntotal, reranker.model_name
            )
            if reranker is not None else None
        )
        self.multicrop_index = crop_state[0] if crop_state is not None else None
        self.multicrop_manifest = crop_state[1] if crop_state is not None else None
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
               quality: bool = True, use_ocr: bool = True,
               use_hybrid: bool = True,
               use_crops: bool = False) -> list[dict[str, object]]:
        if top_k < 1 or top_k > MAX_ANSWERS:
            raise ValueError(f"top_k must be in [1, {MAX_ANSWERS}]")
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")
        normalized_query = normalize_visual_query(query)
        variants = (
            expand_query(normalized_query)
            if self.query_ensemble else [normalized_query]
        )
        vectors = self.encode_many(variants)
        candidate_k = min(max(candidate_k, top_k), self.index.ntotal)
        _, ids = self.index.search(vectors, candidate_k)
        rankings = [[int(value) for value in row if value >= 0] for row in ids]
        ranked_ids, ranked_scores = fuse_query_rankings(
            rankings, limit=candidate_k
        )
        crop_by_id: dict[int, int] = {}
        siglip_vector = None
        if self.siglip2_index is not None and self.reranker is not None and quality:
            siglip_vector = self.reranker.encode_text(normalized_query)
            _, siglip_values = self.siglip2_index.search(siglip_vector, candidate_k)
            siglip_ids = [int(value) for value in siglip_values[0] if value >= 0]
            ranked_ids, ranked_scores = fuse_query_rankings(
                [ranked_ids, siglip_ids],
                weights=[1.0, SIGLIP2_GLOBAL_WEIGHT],
                limit=candidate_k,
            )
        if (
            use_crops and quality and self.multicrop_index is not None
            and self.reranker is not None
        ):
            if siglip_vector is None:
                siglip_vector = self.reranker.encode_text(normalized_query)
            crop_ids, _, crop_by_id = search_multicrop(
                self.multicrop_index, siglip_vector, candidate_k
            )
            ranked_ids, ranked_scores = fuse_query_rankings(
                [ranked_ids, crop_ids], weights=[1.0, 1.0], limit=candidate_k
            )
        top_labels: list[str] = []
        if self.hybrid is not None and use_hybrid:
            text_vector = self.hybrid.encode(normalized_query)
            object_ids, matched_labels = (
                self.hybrid.object_ranking(text_vector, self.hybrid_config)
                if self.hybrid_config.object_weight > 0.0 else ([], [])
            )
            conjunction_ids, conjunction_labels = self.hybrid.object_conjunction_ranking(
                normalized_query, self.hybrid_config
            )
            if conjunction_ids:
                ranked_ids, ranked_scores = fuse_query_rankings(
                    [ranked_ids, conjunction_ids],
                    weights=[1.0, self.hybrid_config.conjunction_weight],
                    limit=candidate_k,
                )
            ocr_ids = (
                self.ocr.ranking(normalized_query)
                if (use_ocr and self.ocr is not None
                    and has_ocr_intent(normalized_query))
                else []
            )
            union_ids = list(dict.fromkeys([*ranked_ids, *object_ids, *ocr_ids]))
            metadata = load_metadata(self.metadata_path, union_ids)
            video_by_id = {
                global_id: str(row["video_id"]) for global_id, row in metadata.items()
            }
            metadata_ranks = (
                self.hybrid.metadata_video_ranks(text_vector)
                if self.hybrid_config.metadata_weight > 0.0 else {}
            )
            if not object_ids and not metadata_ranks and ocr_ids:
                score_by_id = dict(zip(ranked_ids, ranked_scores))
                ranked_ids = protect_signal_ids(ranked_ids, ocr_ids)
                ranked_scores = np.asarray(
                    [score_by_id.get(global_id, 0.0) for global_id in ranked_ids],
                    dtype=np.float32,
                )
            else:
                ranked_ids, ranked_scores = reciprocal_rank_fusion(
                    ranked_ids, object_ids, video_by_id,
                    metadata_ranks, self.hybrid_config,
                    ocr_ids=ocr_ids,
                )
            top_labels = [*conjunction_labels, *[
                label for label, _ in matched_labels[:5]
            ]]
            ocr_ranks = {
                global_id: rank for rank, global_id in enumerate(ocr_ids, start=1)
            }
        else:
            metadata = load_metadata(self.metadata_path, ranked_ids)
            ocr_ranks = {}
        use_image_rerank = (
            self.reranker is not None
            and quality
            and self.siglip2_index is None
        )
        selection_size = (
            max(top_k, self.reranker.config.pool_size)
            if use_image_rerank else top_k
        )
        selected = select_candidates(
            ranked_ids, ranked_scores, metadata, selection_size, per_video, min_time_gap,
            video_pool_limit=(max(1, selection_size // per_video)
                              if use_image_rerank else None),
        )
        for row in selected:
            row["matched_objects"] = top_labels
            global_id = int(row["_global_id"])
            if global_id in crop_by_id:
                row["crop_index"] = crop_by_id[global_id]
            row["_ocr_rank"] = ocr_ranks.get(
                global_id, len(ocr_ranks) + 1
            )
        if use_image_rerank:
            selected = self.reranker.rerank(query, selected)
            selected = diversify_ranked_rows(selected, top_k, unique_prefix=5,
                                             per_video_limit=per_video)
            selected = protect_signal_rows(selected)
        elif quality:
            selected = protect_signal_rows(selected)
        for row in selected:
            row.pop("_global_id", None)
            row.pop("_ocr_rank", None)
        return selected[:top_k]

def create_app(index_dir: Path | None = None, zip_dir: Path | None = None,
               device: str = "auto", engine: object | None = None,
               keyframes: object | None = None, rerank: bool = True,
               reranker_model: str = SIGLIP2_MODEL,
               query_ensemble: bool = False,
               query_package: object | None = None,
               ground_truth: object | None = None) -> Flask:
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
    gt_store = ground_truth or GroundTruthStore(Path("ground_truth/local.jsonl"))
    auto_lock = threading.Lock()
    auto_state: dict[str, object] = {
        "running": False, "completed": 0, "total": 0,
        "current": None, "errors": {},
    }

    def auto_snapshot() -> dict[str, object]:
        with auto_lock:
            return dict(auto_state)

    def run_package_automatically(names: list[str]) -> None:
        errors: dict[str, str] = {}
        for name in names:
            with auto_lock:
                auto_state["current"] = name
            try:
                query = next(item for item in package.status() if item["name"] == name)
                result = assistant.run(
                    str(query["task"]), str(query["text"]), top_k=100,
                    candidate_k=10000, per_video=5, min_time_gap=1.5,
                    quality=getattr(search_engine, "reranker", None) is not None,
                    qa_candidates=10,
                )
                package.save(name, list(result["results"]), human_reviewed=False)
            except Exception as error:
                errors[name] = str(error)
            finally:
                with auto_lock:
                    auto_state["completed"] = int(auto_state["completed"]) + 1
                    auto_state["errors"] = dict(errors)
        with auto_lock:
            auto_state.update(running=False, current=None, errors=dict(errors))


    @app.errorhandler(ValueError)
    def handle_value_error(error: ValueError) -> tuple[Response, int]:
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(MemoryError)
    def handle_memory_error(error: MemoryError) -> tuple[Response, int]:
        app.logger.exception("Request ran out of memory")
        return jsonify({
            "error": (
                "Máy đang thiếu bộ nhớ trong lúc tạo index. "
                "Hãy thử lại sau khi multi-crop hoàn tất."
            )
        }), 503

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
            "ocr_frames": (getattr(search_engine, "ocr", None).count()
                           if getattr(search_engine, "ocr", None) is not None else 0),
            "query_ensemble": getattr(search_engine, "query_ensemble", False),
            "siglip2_vectors": int(getattr(
                getattr(search_engine, "siglip2_index", None), "ntotal", 0
            )),
            "siglip2_crop_vectors": int(getattr(
                getattr(search_engine, "multicrop_index", None), "ntotal", 0
            )),
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
        search_args = [
            query, int(payload.get("top_k", 50)),
            int(payload.get("candidate_k", 5000)), int(payload.get("per_video", 3)),
            float(payload.get("min_time_gap", 2.0)), use_quality,
        ]
        if "use_ocr" in payload or "use_hybrid" in payload:
            search_args.extend([
                bool(payload.get("use_ocr", True)),
                bool(payload.get("use_hybrid", True)),
            ])
        with assistant.lock:
            results = search_engine.search(*search_args)
        return jsonify({"query": query, "count": len(results),
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "mode": "quality" if use_quality else "fast",
            "planner": (
                (("query ensemble + " if getattr(search_engine, "query_ensemble", False) else "")
                 + "clip + objects + metadata"
                 + (" + ocr" if getattr(search_engine, "ocr", None) else "")
                 + " -> rrf"
                 + (" + siglip2-global" if use_quality and getattr(
                     search_engine, "siglip2_index", None) is not None else "")
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
            qa_candidates=int(payload.get("qa_candidates", 10)),
            trake_verify=bool(payload.get(
                "trake_verify",
                bool(payload.get("quality", True)) and torch.cuda.is_available(),
            )),
            trake_candidates=int(payload.get("trake_candidates", 8)),
        ))
    @app.post("/api/qa/candidates")
    def api_qa_candidates() -> Response:
        payload = request.get_json(force=True)
        question = str(payload.get("question", "")).strip()
        if not question:
            raise ValueError("Question must not be empty")
        with assistant.lock:
            rows = assistant._qa_candidate_rows(question)
            rows = assistant._qa_submission_rows(
                rows, int(payload.get("qa_candidates", 10))
            )
        return jsonify({"count": len(rows), "results": rows})
    @app.post("/api/qa/answer")
    def api_qa_answer() -> Response:
        payload = request.get_json(force=True)
        selections = payload.get("selections", [])
        if not isinstance(selections, list):
            raise ValueError("selections must be a list")
        return jsonify(assistant.answer_selected(
            str(payload.get("question", "")), selections
        ))
    @app.get("/api/ground-truth")
    def api_ground_truth_status() -> Response:
        records = gt_store.records()
        return jsonify({"count": len(records), "records": records})

    @app.post("/api/ground-truth")
    def api_ground_truth_save() -> Response:
        record = gt_store.upsert(request.get_json(force=True))
        return jsonify({"record": record, "count": len(gt_store.records())})
    @app.post("/api/package/import")
    def api_package_import() -> Response:
        if bool(auto_snapshot()["running"]):
            raise ValueError("Wait for the automatic package run to finish before importing")
        uploaded = request.files.get("package")
        if uploaded is None:
            raise ValueError("Missing package ZIP")
        return jsonify({"queries": package.import_zip(uploaded.read())})

    @app.get("/api/package/status")
    def api_package_status() -> Response:
        return jsonify({"queries": package.status()})

    @app.post("/api/package/type")
    def api_package_type() -> Response:
        if bool(auto_snapshot()["running"]):
            raise ValueError("Wait for the automatic package run to finish before changing task types")
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
        saved = package.save(
            str(payload.get("query_name", "")), rows,
            human_reviewed=bool(payload.get("human_reviewed", False)),
        )
        return jsonify({"query": saved, "queries": package.status()})

    @app.get("/api/package/review")
    def api_package_review_queue() -> Response:
        return jsonify({"queries": package.review_queue()})

    @app.post("/api/package/review")
    def api_package_review_done() -> Response:
        payload = request.get_json(force=True)
        reviewed = package.mark_reviewed(str(payload.get("query_name", "")))
        return jsonify({"query": reviewed, "queries": package.status()})

    @app.get("/api/package/auto-status")
    def api_package_auto_status() -> Response:
        return jsonify({"auto": auto_snapshot(), "queries": package.status()})

    @app.post("/api/package/auto-run")
    def api_package_auto_run() -> Response:
        payload = request.get_json(silent=True) or {}
        rerun = bool(payload.get("rerun", False))
        names = [
            str(item["name"]) for item in package.status()
            if rerun or not bool(item.get("completed"))
        ]
        if not names:
            raise ValueError("All imported queries already have results")
        with auto_lock:
            if bool(auto_state["running"]):
                raise ValueError("Automatic package run is already active")
            auto_state.update(
                running=True, completed=0, total=len(names),
                current=None, errors={},
            )
        threading.Thread(
            target=run_package_automatically, args=(names,), daemon=True,
            name="aic-package-auto",
        ).start()
        return jsonify({"auto": auto_snapshot()}), 202

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

