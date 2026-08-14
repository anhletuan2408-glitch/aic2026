from __future__ import annotations

import argparse
import io
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from rerank import SIGLIP2_MODEL
from web_app import KeyframeStore


@dataclass(frozen=True)
class FrameRecord:
    global_id: int
    video_id: str
    keyframe_no: int


def load_frame_records(metadata_path: Path) -> list[FrameRecord]:
    with sqlite3.connect(metadata_path) as connection:
        rows = connection.execute(
            "SELECT global_id,video_id,keyframe_no FROM keyframes ORDER BY global_id"
        ).fetchall()
    records = [
        FrameRecord(int(global_id), str(video_id), int(keyframe_no))
        for global_id, video_id, keyframe_no in rows
    ]
    validate_contiguous_global_ids(records)
    return records


def validate_contiguous_global_ids(records: list[FrameRecord]) -> None:
    if not records:
        raise ValueError("Metadata contains no keyframes")
    for expected, record in enumerate(records):
        if record.global_id != expected:
            raise ValueError(
                "SigLIP2 vector storage requires contiguous global_id values; "
                f"expected {expected}, got {record.global_id}"
            )


def _tensor(value: object) -> torch.Tensor:
    return getattr(value, "pooler_output", value)


def encode_image_batch(
    records: list[FrameRecord],
    store: KeyframeStore,
    processor: Any,
    model: Any,
    device: str,
) -> np.ndarray:
    images: list[Image.Image] = []
    try:
        for record in records:
            image = Image.open(
                io.BytesIO(store.get_bytes(record.video_id, record.keyframe_no))
            ).convert("RGB")
            images.append(image)
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            features = _tensor(model.get_image_features(**inputs)).float()
            features /= features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return np.ascontiguousarray(features.cpu().numpy(), dtype=np.float32)
    finally:
        for image in images:
            image.close()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def create_state(
    output_dir: Path,
    frame_count: int,
    dimension: int,
    model_name: str,
) -> tuple[np.memmap, np.memmap, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.lib.format.open_memmap(
        output_dir / "vectors.f16.npy", mode="w+",
        dtype=np.float16, shape=(frame_count, dimension),
    )
    completed = np.lib.format.open_memmap(
        output_dir / "completed.npy", mode="w+",
        dtype=np.bool_, shape=(frame_count,),
    )
    completed[:] = False
    manifest: dict[str, object] = {
        "model": model_name,
        "frames": frame_count,
        "dimension": dimension,
        "dtype": "float16",
        "completed": 0,
        "index_frames": 0,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return vectors, completed, manifest


def load_state(
    output_dir: Path,
    frame_count: int,
    model_name: str,
) -> tuple[np.memmap, np.memmap, dict[str, object]] | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("model")) != model_name:
        raise ValueError(
            f"Existing SigLIP2 state uses {manifest.get('model')}, not {model_name}"
        )
    if int(manifest.get("frames", -1)) != frame_count:
        raise ValueError("Existing SigLIP2 state frame count does not match metadata")
    vectors = np.load(output_dir / "vectors.f16.npy", mmap_mode="r+")
    completed = np.load(output_dir / "completed.npy", mmap_mode="r+")
    expected = (frame_count, int(manifest["dimension"]))
    if vectors.shape != expected or completed.shape != (frame_count,):
        raise ValueError("Existing SigLIP2 state has invalid array shapes")
    return vectors, completed, manifest


def write_faiss_index(
    vectors: np.ndarray,
    completed: np.ndarray,
    output_path: Path,
    chunk_size: int = 4096,
) -> int:
    ids = np.flatnonzero(completed).astype(np.int64)
    if not len(ids):
        raise ValueError("Cannot build an empty SigLIP2 FAISS index")
    dimension = int(vectors.shape[1])
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
    for offset in range(0, len(ids), chunk_size):
        chunk_ids = ids[offset : offset + chunk_size]
        chunk = np.ascontiguousarray(vectors[chunk_ids], dtype=np.float32)
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        chunk /= np.clip(norms, 1e-12, None)
        index.add_with_ids(chunk, chunk_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    faiss.write_index(index, str(temporary))
    temporary.replace(output_path)
    return int(index.ntotal)


def build(args: argparse.Namespace) -> None:
    records = load_frame_records(args.metadata)
    state = load_state(args.output_dir, len(records), args.model)
    completed_before = (
        int(np.count_nonzero(state[1])) if state is not None else 0
    )
    pending = [
        record for record in records
        if state is None or not bool(state[1][record.global_id])
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    processor = AutoProcessor.from_pretrained(args.model)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    store = KeyframeStore(args.zip_dir)
    started = time.perf_counter()
    processed = 0
    vectors: np.memmap
    completed: np.memmap
    manifest: dict[str, object]
    try:
        if state is not None:
            vectors, completed, manifest = state
        elif pending:
            first = pending[: args.batch_size]
            first_vectors = encode_image_batch(
                first, store, processor, model, args.device
            )
            vectors, completed, manifest = create_state(
                args.output_dir, len(records), int(first_vectors.shape[1]), args.model
            )
            first_ids = np.asarray([row.global_id for row in first], dtype=np.int64)
            vectors[first_ids] = first_vectors.astype(np.float16)
            completed[first_ids] = True
            processed += len(first)
            pending = pending[len(first) :]
        else:
            raise RuntimeError("No pending frames and no existing SigLIP2 state")

        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            features = encode_image_batch(
                batch, store, processor, model, args.device
            )
            ids = np.asarray([row.global_id for row in batch], dtype=np.int64)
            vectors[ids] = features.astype(np.float16)
            completed[ids] = True
            processed += len(batch)
            if processed % args.flush_every < len(batch):
                vectors.flush()
                completed.flush()
                total = completed_before + processed
                elapsed = max(time.perf_counter() - started, 1e-6)
                print(
                    f"SigLIP2 {total}/{len(records)} "
                    f"({processed / elapsed:.2f} frame/s)",
                    flush=True,
                )
        vectors.flush()
        completed.flush()
        manifest["completed"] = int(np.count_nonzero(completed))
        manifest["seconds_last_run"] = round(time.perf_counter() - started, 3)
        should_finalize = bool(np.all(completed)) or args.finalize_partial
        if should_finalize:
            manifest["index_frames"] = write_faiss_index(
                vectors, completed, args.output_dir / "keyframes.faiss"
            )
        atomic_write_json(args.output_dir / "manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    finally:
        store.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a resumable global SigLIP2 keyframe FAISS index"
    )
    parser.add_argument("--metadata", type=Path, default=Path("index/metadata.sqlite3"))
    parser.add_argument("--zip-dir", type=Path, default=Path("E:/"))
    parser.add_argument("--output-dir", type=Path, default=Path("index/siglip2"))
    parser.add_argument("--model", default=SIGLIP2_MODEL)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--flush-every", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--finalize-partial", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 128:
        parser.error("--batch-size must be in [1, 128]")
    if args.flush_every < args.batch_size:
        parser.error("--flush-every must be at least batch-size")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    build(parse_args())
