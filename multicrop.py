from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from PIL import Image


CROP_BOXES: tuple[tuple[float, float, float, float], ...] = (
    (0.0, 0.0, 0.6, 0.6),
    (0.4, 0.0, 1.0, 0.6),
    (0.0, 0.4, 0.6, 1.0),
    (0.4, 0.4, 1.0, 1.0),
    (0.2, 0.2, 0.8, 0.8),
)
CROP_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right", "center")


def crop_image(image: Image.Image, crop_index: int) -> Image.Image:
    if crop_index < 0 or crop_index >= len(CROP_BOXES):
        raise ValueError(f"crop_index must be in [0, {len(CROP_BOXES) - 1}]")
    width, height = image.size
    left, top, right, bottom = CROP_BOXES[crop_index]
    box = (
        round(left * width), round(top * height),
        round(right * width), round(bottom * height),
    )
    return image.crop(box)


def encode_crop_id(global_id: int, crop_index: int,
                   crop_count: int = len(CROP_BOXES)) -> int:
    if global_id < 0 or not 0 <= crop_index < crop_count:
        raise ValueError("Invalid global_id or crop_index")
    return global_id * crop_count + crop_index


def decode_crop_id(crop_id: int,
                   crop_count: int = len(CROP_BOXES)) -> tuple[int, int]:
    if crop_id < 0 or crop_count < 1:
        raise ValueError("Invalid crop_id or crop_count")
    return divmod(crop_id, crop_count)


def collapse_crop_results(
    crop_ids: np.ndarray, scores: np.ndarray, limit: int,
    crop_count: int = len(CROP_BOXES),
) -> tuple[list[int], np.ndarray, dict[int, int]]:
    """Collapse crop hits to frames, retaining each frame's strongest region."""
    if limit < 1 or len(crop_ids) != len(scores):
        raise ValueError("Invalid crop search results")
    frame_ids: list[int] = []
    frame_scores: list[float] = []
    best_crops: dict[int, int] = {}
    for raw_id, raw_score in zip(crop_ids, scores):
        crop_id = int(raw_id)
        if crop_id < 0:
            continue
        global_id, crop_index = decode_crop_id(crop_id, crop_count)
        if global_id in best_crops:
            continue
        best_crops[global_id] = crop_index
        frame_ids.append(global_id)
        frame_scores.append(float(raw_score))
        if len(frame_ids) >= limit:
            break
    return (
        frame_ids,
        np.asarray(frame_scores, dtype=np.float32),
        best_crops,
    )


def load_multicrop_index(
    directory: Path, expected_frames: int, model_name: str,
) -> tuple[Any, dict[str, Any]] | None:
    manifest_path = directory / "manifest.json"
    index_path = directory / "crops.faiss"
    if not manifest_path.exists() and not index_path.exists():
        return None
    if not manifest_path.exists():
        raise ValueError("Multi-crop index is missing manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    crop_count = int(manifest.get("crop_count", -1))
    frames = int(manifest.get("frames", -1))
    completed = int(manifest.get("completed", -1))
    index_vectors = int(manifest.get("index_vectors", -1))
    if (
        str(manifest.get("model")) != model_name
        or crop_count != len(CROP_BOXES)
        or frames != expected_frames
    ):
        raise ValueError("Multi-crop manifest does not match runtime configuration")
    if completed < frames and index_vectors <= 0:
        return None
    expected_vectors = frames * crop_count
    if completed != frames or index_vectors != expected_vectors or not index_path.exists():
        raise ValueError("Multi-crop index is partial or incomplete")
    index = faiss.read_index(str(index_path))
    if index.ntotal != expected_vectors or index.d != int(manifest["dimension"]):
        raise ValueError("Multi-crop FAISS dimensions/count do not match manifest")
    return index, manifest


def search_multicrop(
    index: Any, query_vector: np.ndarray, frame_limit: int,
    crop_count: int = len(CROP_BOXES),
) -> tuple[list[int], np.ndarray, dict[int, int]]:
    depth = min(int(index.ntotal), frame_limit * crop_count)
    scores, ids = index.search(
        np.ascontiguousarray(query_vector, dtype=np.float32), depth
    )
    return collapse_crop_results(ids[0], scores[0], frame_limit, crop_count)
