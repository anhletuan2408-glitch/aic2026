from __future__ import annotations

import json
import re
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class HybridConfig:
    rrf_k: int = 60
    object_weight: float = 0.0
    metadata_weight: float = 0.0
    temporal_weight: float = 0.20
    ocr_weight: float = 1.10
    object_labels: int = 12
    object_candidates: int = 2500
    conjunction_weight: float = 0.45


def _fold_query(value: str) -> str:
    folded = unicodedata.normalize("NFD", value.casefold())
    return "".join(
        char for char in folded if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")


def object_concept_groups(query: str) -> list[tuple[str, ...]]:
    """Map explicit query entities to detector-label groups, preserving AND intent."""
    text = _fold_query(query)
    groups: list[tuple[str, ...]] = []
    if re.search(r"\b(?:phu nu|nguoi nu|woman|female)\b", text):
        groups.append(("Woman", "Person"))
    elif re.search(r"\b(?:dan ong|nguoi nam|man|male)\b", text):
        groups.append(("Man", "Person"))
    elif re.search(r"\b(?:nguoi|person|people)\b", text):
        groups.append(("Person",))
    concepts = (
        (r"\b(?:xe may|mo to|motorcycle|motorbike)\b", ("Motorcycle",)),
        (r"\b(?:xe dap|bicycle|bike)\b", ("Bicycle",)),
        (r"\b(?:o to|xe hoi|car)\b", ("Car",)),
        (r"\b(?:xe buyt|bus)\b", ("Bus",)),
        (r"\b(?:xe tai|truck)\b", ("Truck",)),
        (r"\b(?:con meo|meo|cat)\b", ("Cat",)),
        (r"\b(?:con cho|cho|dog)\b", ("Dog",)),
        (r"\b(?:ghe|chair|seat)\b", ("Chair", "Couch", "Bench")),
        (r"\b(?:cua|door)\b", ("Door",)),
        (r"\b(?:micro|microphone)\b", ("Microphone",)),
    )
    for pattern, labels in concepts:
        if re.search(pattern, text):
            groups.append(labels)
    return groups


class HybridSignals:
    def __init__(self, directory: Path, device: str = "cpu") -> None:
        manifest = json.loads((directory / "hybrid_manifest.json").read_text())
        self.text_model_name = str(manifest["text_model"])
        self.labels = json.loads((directory / "object_labels.json").read_text("utf-8"))
        data = np.load(directory / "hybrid_signals.npz", allow_pickle=False)
        self.offsets = data["offsets"]
        self.frame_ids = data["frame_ids"]
        self.scores = data["scores"].astype(np.float32)
        self.label_vectors = data["label_vectors"].astype(np.float32)
        self.label_idf = data["label_idf"].astype(np.float32)
        self.video_ids = [str(value) for value in data["video_ids"]]
        self.video_vectors = data["video_vectors"].astype(np.float32)
        self.frame_count = int(data["frame_count"][0])
        self._label_by_name = {
            label.casefold(): index for index, label in enumerate(self.labels)
        }
        self.model = SentenceTransformer(self.text_model_name, device=device)
        self._model_lock = threading.Lock()

    def encode(self, query: str) -> np.ndarray:
        return self.encode_many([query])[0]

    def encode_many(self, queries: list[str]) -> np.ndarray:
        with self._model_lock:
            vectors = self.model.encode(
                [f"query: {query}" for query in queries],
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return vectors.astype(np.float32)

    def object_ranking(
        self, query_vector: np.ndarray, config: HybridConfig
    ) -> tuple[list[int], list[tuple[str, float]]]:
        similarities = self.label_vectors @ query_vector
        top_labels = np.argsort(similarities)[-config.object_labels :][::-1]
        selected_sims = similarities[top_labels]
        relevance = np.exp((selected_sims - selected_sims.max()) * 18.0)
        relevance *= np.clip(self.label_idf[top_labels], 1.0, 8.0)
        relevance /= max(float(relevance.max()), 1e-8)
        frame_scores = np.zeros(self.frame_count, dtype=np.float32)
        explanations: list[tuple[str, float]] = []
        for label_id, weight in zip(top_labels, relevance):
            start, end = self.offsets[label_id : label_id + 2]
            ids = self.frame_ids[start:end]
            values = self.scores[start:end]
            np.maximum.at(frame_scores, ids, values * float(weight))
            explanations.append((self.labels[label_id], float(selected_sims[len(explanations)])))
        count = min(config.object_candidates, int(np.count_nonzero(frame_scores)))
        if not count:
            return [], explanations
        ids = np.argpartition(frame_scores, -count)[-count:]
        ids = ids[np.argsort(frame_scores[ids])[::-1]]
        return [int(value) for value in ids], explanations

    def object_conjunction_ranking(
        self, query: str, config: HybridConfig
    ) -> tuple[list[int], list[str]]:
        """Rank frames that contain every distinct object concept in the query."""
        requested = object_concept_groups(query)
        if len(requested) < 2:
            return [], []
        group_scores: list[np.ndarray] = []
        matched: list[str] = []
        for alternatives in requested:
            scores = np.zeros(self.frame_count, dtype=np.float32)
            found = []
            for label in alternatives:
                label_id = self._label_by_name.get(label.casefold())
                if label_id is None:
                    continue
                start, end = self.offsets[label_id:label_id + 2]
                ids = self.frame_ids[start:end]
                np.maximum.at(scores, ids, self.scores[start:end])
                found.append(label)
            if not found:
                continue
            group_scores.append(scores)
            matched.append("/".join(found))
        if len(group_scores) < 2:
            return [], matched
        stacked = np.stack(group_scores)
        valid = np.all(stacked > 0.0, axis=0)
        ids = np.flatnonzero(valid)
        if not len(ids):
            return [], matched
        scores = stacked[:, ids].min(axis=0) + 0.25 * stacked[:, ids].mean(axis=0)
        count = min(config.object_candidates, len(ids))
        selected = np.argpartition(scores, -count)[-count:]
        selected = selected[np.argsort(scores[selected])[::-1]]
        return [int(ids[index]) for index in selected], matched

    def metadata_video_ranks(self, query_vector: np.ndarray) -> dict[str, int]:
        similarities = self.video_vectors @ query_vector
        order = np.argsort(similarities)[::-1]
        return {self.video_ids[index]: rank for rank, index in enumerate(order, start=1)}


def reciprocal_rank_fusion(
    base_ids: list[int],
    object_ids: list[int],
    video_by_id: dict[int, str],
    metadata_ranks: dict[str, int],
    config: HybridConfig,
    temporal_ids: list[int] | None = None,
    ocr_ids: list[int] | None = None,
) -> tuple[list[int], np.ndarray]:
    scores: dict[int, float] = {}
    for rank, global_id in enumerate(base_ids, start=1):
        scores[global_id] = scores.get(global_id, 0.0) + 1.0 / (config.rrf_k + rank)
    for rank, global_id in enumerate(object_ids, start=1):
        scores[global_id] = scores.get(global_id, 0.0) + config.object_weight / (
            config.rrf_k + rank
        )
    for rank, global_id in enumerate(temporal_ids or [], start=1):
        scores[global_id] = scores.get(global_id, 0.0) + config.temporal_weight / (
            config.rrf_k + rank
        )
    for rank, global_id in enumerate(ocr_ids or [], start=1):
        scores[global_id] = scores.get(global_id, 0.0) + config.ocr_weight / (
            config.rrf_k + rank
        )

    for global_id in list(scores):
        video_rank = metadata_ranks.get(video_by_id.get(global_id, ""))
        if video_rank is not None:
            scores[global_id] += config.metadata_weight / (config.rrf_k + video_rank)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return ordered, np.asarray([scores[value] for value in ordered], dtype=np.float32)
