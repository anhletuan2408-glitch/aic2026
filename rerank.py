from __future__ import annotations

import io
import threading
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


SIGLIP2_BASE_MODEL = "google/siglip2-base-patch16-224"
SIGLIP2_LARGE_MODEL = "google/siglip2-large-patch16-384"
SIGLIP2_MODEL = SIGLIP2_LARGE_MODEL


@dataclass(frozen=True)
class RerankConfig:
    batch_size: int = 32
    weight: float = 0.8
    rrf_k: int = 60
    pool_size: int = 300


def fuse_rerank_scores(
    rows: list[dict[str, object]], similarities: np.ndarray, config: RerankConfig
) -> list[dict[str, object]]:
    order = np.argsort(similarities)[::-1]
    rerank_positions = np.empty(len(rows), dtype=np.int32)
    rerank_positions[order] = np.arange(1, len(rows) + 1)
    output = []
    for base_rank, (row, rerank_rank, similarity) in enumerate(
        zip(rows, rerank_positions, similarities), start=1
    ):
        fused = 1.0 / (config.rrf_k + base_rank) + config.weight / (
            config.rrf_k + int(rerank_rank)
        )
        output.append(
            {
                **row,
                "pre_rerank_score": float(row["score"]),
                "siglip2_score": float(similarity),
                "score": fused,
            }
        )
    output.sort(key=lambda row: float(row["score"]), reverse=True)
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


class Siglip2Reranker:
    def __init__(self, keyframes: object, device: str = "cuda",
                 config: RerankConfig | None = None,
                 model_name: str = SIGLIP2_MODEL) -> None:
        self.keyframes = keyframes
        self.device = device
        self.config = config or RerankConfig()
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype).to(device).eval()
        self._lock = threading.Lock()

    @staticmethod
    def _tensor(value: object) -> torch.Tensor:
        return getattr(value, "pooler_output", value)

    def rerank(self, query: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        with self._lock:
            return self._rerank_locked(query, rows)

    def encode_text(self, query: str) -> np.ndarray:
        """Encode one query for the precomputed global SigLIP2 index."""
        return self.encode_text_many([query])

    def encode_text_many(self, queries: list[str]) -> np.ndarray:
        queries = [query.strip() for query in queries]
        if not queries or any(not query for query in queries):
            raise ValueError("Queries must not be empty")
        with self._lock:
            features = self._encode_text_locked(queries)
        return np.ascontiguousarray(features.float().cpu().numpy(), dtype=np.float32)

    def _encode_text_locked(self, queries: list[str]) -> torch.Tensor:
        inputs = self.processor(
            text=queries, padding="max_length", return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            features = self._tensor(self.model.get_text_features(**inputs)).float()
            features /= features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features

    def _rerank_locked(self, query: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        if not rows:
            return []
        text = self._encode_text_locked([query])
        all_scores: list[np.ndarray] = []
        for offset in range(0, len(rows), self.config.batch_size):
            batch = rows[offset : offset + self.config.batch_size]
            images = [
                Image.open(
                    io.BytesIO(
                        self.keyframes.get_bytes(
                            str(row["video_id"]), int(row["keyframe_no"])
                        )
                    )
                ).convert("RGB")
                for row in batch
            ]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                features = self._tensor(
                    self.model.get_image_features(**inputs)
                ).float()
                features = features / features.norm(dim=-1, keepdim=True)
                scores = (features @ text.T).squeeze(1).float().cpu().numpy()
            all_scores.append(scores)
            for image in images:
                image.close()
        return fuse_rerank_scores(rows, np.concatenate(all_scores), self.config)
