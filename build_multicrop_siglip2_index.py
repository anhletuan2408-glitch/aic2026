from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from build_siglip2_index import (
    FrameRecord, atomic_write_json, load_frame_records,
)
from multicrop import CROP_BOXES, CROP_NAMES, crop_image, encode_crop_id
from rerank import SIGLIP2_MODEL
from web_app import KeyframeStore


def _tensor(value: object) -> torch.Tensor:
    return getattr(value, "pooler_output", value)


def encode_crop_batch(
    records: list[FrameRecord], store: KeyframeStore, processor: Any,
    model: Any, device: str,
) -> np.ndarray:
    images: list[Image.Image] = []
    sources: list[Image.Image] = []
    try:
        for record in records:
            source = Image.open(
                io.BytesIO(store.get_bytes(record.video_id, record.keyframe_no))
            ).convert("RGB")
            sources.append(source)
            images.extend(crop_image(source, index) for index in range(len(CROP_BOXES)))
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            features = _tensor(model.get_image_features(**inputs)).float()
            features /= features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return np.ascontiguousarray(
            features.cpu().numpy().reshape(len(records), len(CROP_BOXES), -1),
            dtype=np.float32,
        )
    finally:
        for image in images:
            image.close()
        for source in sources:
            source.close()


def create_state(output_dir: Path, frame_count: int, dimension: int,
                 model_name: str) -> tuple[np.memmap, np.memmap, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.lib.format.open_memmap(
        output_dir / "vectors.f16.npy", mode="w+", dtype=np.float16,
        shape=(frame_count, len(CROP_BOXES), dimension),
    )
    completed = np.lib.format.open_memmap(
        output_dir / "completed.npy", mode="w+", dtype=np.bool_,
        shape=(frame_count,),
    )
    completed[:] = False
    manifest: dict[str, Any] = {
        "model": model_name,
        "frames": frame_count,
        "crop_count": len(CROP_BOXES),
        "crop_names": list(CROP_NAMES),
        "crop_boxes": [list(box) for box in CROP_BOXES],
        "dimension": dimension,
        "dtype": "float16",
        "completed": 0,
        "index_vectors": 0,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return vectors, completed, manifest


def load_state(output_dir: Path, frame_count: int, model_name: str):
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("model")) != model_name:
        raise ValueError("Existing crop state uses a different model")
    if (
        int(manifest.get("frames", -1)) != frame_count
        or int(manifest.get("crop_count", -1)) != len(CROP_BOXES)
    ):
        raise ValueError("Existing crop state does not match metadata/crop layout")
    vectors = np.load(output_dir / "vectors.f16.npy", mmap_mode="r+")
    completed = np.load(output_dir / "completed.npy", mmap_mode="r+")
    expected = (frame_count, len(CROP_BOXES), int(manifest["dimension"]))
    if vectors.shape != expected or completed.shape != (frame_count,):
        raise ValueError("Existing crop state has invalid array shapes")
    return vectors, completed, manifest


def write_faiss_index(vectors: np.ndarray, completed: np.ndarray,
                      output_path: Path, chunk_frames: int = 512) -> int:
    frame_ids = np.flatnonzero(completed).astype(np.int64)
    if not len(frame_ids):
        raise ValueError("Cannot build an empty crop index")
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(int(vectors.shape[2])))
    for offset in range(0, len(frame_ids), chunk_frames):
        ids = frame_ids[offset: offset + chunk_frames]
        chunk = np.ascontiguousarray(vectors[ids].reshape(-1, vectors.shape[2]),
                                     dtype=np.float32)
        chunk /= np.clip(np.linalg.norm(chunk, axis=1, keepdims=True), 1e-12, None)
        crop_ids = np.asarray([
            encode_crop_id(int(global_id), crop_index)
            for global_id in ids for crop_index in range(len(CROP_BOXES))
        ], dtype=np.int64)
        index.add_with_ids(chunk, crop_ids)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    faiss.write_index(index, str(temporary))
    temporary.replace(output_path)
    return int(index.ntotal)


def build(args: argparse.Namespace) -> None:
    records = load_frame_records(args.metadata)
    state = load_state(args.output_dir, len(records), args.model)
    pending = [
        record for record in records
        if state is None or not bool(state[1][record.global_id])
    ]
    if args.limit is not None:
        pending = pending[:args.limit]
    processor = AutoProcessor.from_pretrained(args.model)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    store = KeyframeStore(args.zip_dir)
    started = time.perf_counter()
    processed = 0
    try:
        if state is not None:
            vectors, completed, manifest = state
        elif pending:
            first = pending[:args.batch_size]
            encoded = encode_crop_batch(first, store, processor, model, args.device)
            vectors, completed, manifest = create_state(
                args.output_dir, len(records), int(encoded.shape[2]), args.model
            )
            ids = np.asarray([row.global_id for row in first], dtype=np.int64)
            vectors[ids] = encoded.astype(np.float16)
            completed[ids] = True
            processed += len(first)
            pending = pending[len(first):]
        else:
            raise RuntimeError("No pending frames and no existing crop state")
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset: offset + args.batch_size]
            encoded = encode_crop_batch(batch, store, processor, model, args.device)
            ids = np.asarray([row.global_id for row in batch], dtype=np.int64)
            vectors[ids] = encoded.astype(np.float16)
            completed[ids] = True
            processed += len(batch)
            if processed % args.flush_every < len(batch):
                vectors.flush()
                completed.flush()
                elapsed = max(time.perf_counter() - started, 1e-6)
                print(
                    f"SigLIP2 crops {np.count_nonzero(completed)}/{len(records)} "
                    f"frames ({processed * len(CROP_BOXES) / elapsed:.2f} crops/s)",
                    flush=True,
                )
        vectors.flush()
        completed.flush()
        manifest["completed"] = int(np.count_nonzero(completed))
        manifest["seconds_last_run"] = round(time.perf_counter() - started, 3)
        if bool(np.all(completed)) or args.finalize_partial:
            manifest["index_vectors"] = write_faiss_index(
                vectors, completed, args.output_dir / "crops.faiss"
            )
        atomic_write_json(args.output_dir / "manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    finally:
        store.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a resumable five-region SigLIP2 FAISS index"
    )
    parser.add_argument("--metadata", type=Path, default=Path("index/metadata.sqlite3"))
    parser.add_argument("--zip-dir", type=Path, default=Path("E:/"))
    parser.add_argument("--output-dir", type=Path, default=Path("index/siglip2-crops"))
    parser.add_argument("--model", default=SIGLIP2_MODEL)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--flush-every", type=int, default=120)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--finalize-partial", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 24:
        parser.error("--batch-size must be in [1, 24]")
    if args.flush_every < args.batch_size:
        parser.error("--flush-every must be at least batch-size")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    build(parse_args())
