from __future__ import annotations
import argparse
import io
import threading
import time
from pathlib import Path
import faiss
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from sentence_transformers import SentenceTransformer
from search import choose_device, load_metadata
from search_kis import select_candidates
from submission import MAX_ANSWERS
from web_app import KeyframeStore

MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

class MultilingualFaissEngine:
    def __init__(self, index_dir: Path, device: str = "auto") -> None:
        self.device = choose_device(device)
        self.model_name = MODEL_NAME
        self.index = faiss.read_index(str(index_dir / "keyframes.faiss"))
        self.metadata_path = index_dir / "metadata.sqlite3"
        self.model = SentenceTransformer(MODEL_NAME, device=self.device)
        self._model_lock = threading.Lock()

    def encode(self, query: str) -> np.ndarray:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")
        with self._model_lock:
            vector = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return np.ascontiguousarray(vector, dtype=np.float32)

    def search(self, query: str, top_k: int, candidate_k: int = 5000,
               per_video: int = 3, min_time_gap: float = 2.0) -> list[dict[str, object]]:
        if top_k < 1 or top_k > MAX_ANSWERS:
            raise ValueError(f"top_k must be in [1, {MAX_ANSWERS}]")
        vector = self.encode(query)
        candidate_k = min(max(candidate_k, top_k), self.index.ntotal)
        scores, ids = self.index.search(vector, candidate_k)
        ranked_ids = [int(value) for value in ids[0] if value >= 0]
        metadata = load_metadata(self.metadata_path, ranked_ids)
        return select_candidates(ranked_ids, scores[0], metadata, top_k, per_video, min_time_gap)

def create_app(index_dir: Path | None = None, zip_dir: Path | None = None,
               device: str = "auto", engine: object | None = None,
               keyframes: object | None = None) -> Flask:
    app = Flask(__name__)
    search_engine = engine or MultilingualFaissEngine(index_dir or Path("index"), device)
    keyframe_store = keyframes or KeyframeStore(zip_dir or Path("."))

    @app.errorhandler(ValueError)
    def handle_value_error(error: ValueError) -> tuple[Response, int]:
        return jsonify({"error": str(error)}), 400

    @app.get("/")
    def home() -> str:
        return render_template("index_vi.html")

    @app.get("/api/health")
    def health() -> Response:
        index = getattr(search_engine, "index", None)
        return jsonify({"status": "ready", "model": getattr(search_engine, "model_name", MODEL_NAME),
            "device": getattr(search_engine, "device", "test"), "vectors": int(getattr(index, "ntotal", 0)),
            "videos": int(getattr(keyframe_store, "video_count", 0)),
            "index": "FAISS IndexFlatIP", "language": "Vietnamese"})

    @app.post("/api/search")
    def api_search() -> Response:
        payload = request.get_json(force=True)
        started = time.perf_counter()
        query = str(payload.get("query", ""))
        results = search_engine.search(query, int(payload.get("top_k", 50)),
            int(payload.get("candidate_k", 5000)), int(payload.get("per_video", 3)),
            float(payload.get("min_time_gap", 2.0)))
        return jsonify({"query": query, "count": len(results),
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "planner": "multilingual-clip -> faiss -> diversify", "results": results})

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
    args = parser.parse_args()
    create_app(args.index_dir, args.zip_dir, args.device).run(host=args.host, port=args.port, debug=False, threaded=True)

if __name__ == "__main__":
    main()

