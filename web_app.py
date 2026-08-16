from __future__ import annotations

import argparse
import io
import threading
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable

import faiss
import numpy as np
import open_clip
import torch
from flask import Flask, Response, jsonify, render_template, request

from search import choose_device, load_metadata
from search_kis import select_candidates
from submission import MAX_ANSWERS, validate_video_id


class KeyframeStore:
    def __init__(self, zip_dir: Path) -> None:
        self.zip_dir = zip_dir
        self._lock = threading.RLock()
        self._archives: dict[Path, zipfile.ZipFile] = {}
        self._video_archives: dict[str, Path] = {}
        self._build_index()

    def _build_index(self) -> None:
        zip_paths = sorted(self.zip_dir.glob("Keyframes_*.zip"))
        if not zip_paths:
            raise FileNotFoundError(f"No Keyframes_*.zip files found in {self.zip_dir}")
        for zip_path in zip_paths:
            archive = zipfile.ZipFile(zip_path)
            self._archives[zip_path] = archive
            for info in archive.infolist():
                parts = PurePosixPath(info.filename).parts
                if len(parts) >= 3 and parts[0] == "keyframes":
                    self._video_archives.setdefault(parts[1], zip_path)

    @property
    def video_count(self) -> int:
        return len(self._video_archives)

    def get_bytes(self, video_id: str, keyframe_no: int) -> bytes:
        validate_video_id(video_id)
        if keyframe_no < 1:
            raise ValueError("keyframe_no must be positive")
        zip_path = self._video_archives.get(video_id)
        if zip_path is None:
            raise KeyError(video_id)
        entry = f"keyframes/{video_id}/{keyframe_no:03d}.jpg"
        with self._lock:
            return self._archives[zip_path].read(entry)

    def close(self) -> None:
        with self._lock:
            for archive in self._archives.values():
                archive.close()
            self._archives.clear()


class FaissSearchEngine:
    def __init__(self, index_dir: Path, device: str = "auto") -> None:
        self.index_dir = index_dir
        self.device = choose_device(device)
        self.index = faiss.read_index(str(index_dir / "keyframes.faiss"))
        self.metadata_path = index_dir / "metadata.sqlite3"
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="openai", device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
        self.model.eval()
        self._model_lock = threading.Lock()

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        values = [value.strip() for value in texts if value.strip()]
        if not values:
            raise ValueError("At least one non-blank query text is required")
        with self._model_lock, torch.inference_mode():
            tokens = self.tokenizer(values).to(self.device)
            features = self.model.encode_text(tokens).float()
            features /= features.norm(dim=-1, keepdim=True)
            combined = features.mean(dim=0, keepdim=True)
            combined /= combined.norm(dim=-1, keepdim=True)
        return np.ascontiguousarray(combined.cpu().numpy(), dtype=np.float32)

    def search(
        self,
        query: str,
        variants: list[str],
        top_k: int,
        candidate_k: int = 5000,
        per_video: int = 3,
        min_time_gap: float = 2.0,
    ) -> list[dict[str, object]]:
        if top_k < 1 or top_k > MAX_ANSWERS:
            raise ValueError(f"top_k must be in [1, {MAX_ANSWERS}]")
        query_vector = self.encode([query, *variants])
        candidate_k = min(max(candidate_k, top_k), self.index.ntotal)
        scores, ids = self.index.search(query_vector, candidate_k)
        ranked_ids = [int(value) for value in ids[0] if value >= 0]
        metadata = load_metadata(self.metadata_path, ranked_ids)
        return select_candidates(
            ranked_ids,
            scores[0],
            metadata,
            top_k,
            per_video,
            min_time_gap,
        )


def create_app(
    index_dir: Path | None = None,
    zip_dir: Path | None = None,
    device: str = "auto",
    engine: object | None = None,
    keyframes: object | None = None,
) -> Flask:
    app = Flask(__name__)
    search_engine = engine or FaissSearchEngine(index_dir or Path("index"), device)
    keyframe_store = keyframes or KeyframeStore(zip_dir or Path("."))

    @app.errorhandler(ValueError)
    def handle_value_error(error: ValueError) -> tuple[Response, int]:
        return jsonify({"error": str(error)}), 400

    @app.get("/")
    def home() -> str:
        return render_template("index.html")

    @app.get("/api/health")
    def health() -> Response:
        index = getattr(search_engine, "index", None)
        return jsonify(
            {
                "status": "ok",
                "vectors": int(getattr(index, "ntotal", 0)),
                "videos": int(getattr(keyframe_store, "video_count", 0)),
            }
        )

    @app.post("/api/search")
    def api_search() -> Response:
        payload = request.get_json(force=True)
        query = str(payload.get("query", ""))
        variants = [str(value) for value in payload.get("variants", [])]
        results = search_engine.search(
            query,
            variants,
            int(payload.get("top_k", 50)),
            int(payload.get("candidate_k", 5000)),
            int(payload.get("per_video", 3)),
            float(payload.get("min_time_gap", 2.0)),
        )
        return jsonify({"query": query, "count": len(results), "results": results})

    @app.get("/keyframe/<video_id>/<int:keyframe_no>.jpg")
    def keyframe(video_id: str, keyframe_no: int) -> Response:
        try:
            image = keyframe_store.get_bytes(video_id, keyframe_no)
        except (KeyError, FileNotFoundError):
            return jsonify({"error": "Keyframe not found"}), 404
        return Response(
            io.BytesIO(image).getvalue(),
            mimetype="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local AIC retrieval UI.")
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--zip-dir", type=Path, default=Path(".."))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.index_dir, args.zip_dir, args.device)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
