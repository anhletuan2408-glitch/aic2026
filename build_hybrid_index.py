from __future__ import annotations

import argparse
import json
import sqlite3
from array import array
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sentence_transformers import SentenceTransformer


TEXT_MODEL = "intfloat/multilingual-e5-small"


def load_frame_mapping(metadata_path: Path) -> tuple[dict[tuple[str, int], int], int]:
    with sqlite3.connect(metadata_path) as connection:
        rows = connection.execute(
            "SELECT global_id, video_id, keyframe_no FROM keyframes"
        ).fetchall()
    mapping = {(str(video), int(number)): int(global_id) for global_id, video, number in rows}
    return mapping, len(rows)


def collect_objects(
    objects_zip: Path,
    frame_mapping: dict[tuple[str, int], int],
    min_score: float,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    label_to_id: dict[str, int] = {}
    frame_ids = array("I")
    label_ids = array("H")
    scores = array("f")
    with ZipFile(objects_zip) as archive:
        files = (name for name in archive.namelist() if name.endswith(".json"))
        for name in files:
            parts = Path(name).parts
            video_id, keyframe_no = parts[-2], int(Path(parts[-1]).stem)
            global_id = frame_mapping.get((video_id, keyframe_no))
            if global_id is None:
                continue
            payload = json.loads(archive.read(name))
            best: dict[int, float] = {}
            for label, raw_score in zip(
                payload["detection_class_entities"], payload["detection_scores"]
            ):
                score = float(raw_score)
                if score < min_score:
                    continue
                label_id = label_to_id.setdefault(str(label), len(label_to_id))
                best[label_id] = max(score, best.get(label_id, 0.0))
            for label_id, score in best.items():
                frame_ids.append(global_id)
                label_ids.append(label_id)
                scores.append(score)

    frames = np.frombuffer(frame_ids, dtype=np.uint32).astype(np.int32)
    labels_for_rows = np.frombuffer(label_ids, dtype=np.uint16)
    values = np.frombuffer(scores, dtype=np.float32)
    order = np.argsort(labels_for_rows, kind="stable")
    labels = [""] * len(label_to_id)
    for label, label_id in label_to_id.items():
        labels[label_id] = label
    counts = np.bincount(labels_for_rows, minlength=len(labels))
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return labels, offsets, frames[order], values[order].astype(np.float16)


def load_video_texts(metadata_path: Path) -> tuple[list[str], list[str]]:
    with sqlite3.connect(metadata_path) as connection:
        rows = connection.execute(
            """SELECT video_id, title, description
               FROM keyframes GROUP BY video_id ORDER BY video_id"""
        ).fetchall()
    video_ids, texts = [], []
    for video_id, title, description in rows:
        video_ids.append(str(video_id))
        description = " ".join(str(description).split())[:1200]
        texts.append(f"passage: {title}. {description}")
    return video_ids, texts


def build_index(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping, frame_count = load_frame_mapping(args.metadata)
    labels, offsets, frame_ids, scores = collect_objects(
        args.objects_zip, mapping, args.min_object_score
    )
    model = SentenceTransformer(TEXT_MODEL, device=args.device)
    label_texts = [f"passage: a photo containing {label}" for label in labels]
    label_vectors = model.encode(
        label_texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float16)
    video_ids, video_texts = load_video_texts(args.metadata)
    video_vectors = model.encode(
        video_texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float16)
    document_frequency = np.diff(offsets)
    idf = np.log((frame_count + 1) / (document_frequency + 1)).astype(np.float32)
    np.savez_compressed(
        args.output_dir / "hybrid_signals.npz",
        offsets=offsets,
        frame_ids=frame_ids,
        scores=scores,
        label_vectors=label_vectors,
        label_idf=idf,
        video_ids=np.asarray(video_ids),
        video_vectors=video_vectors,
        frame_count=np.asarray([frame_count], dtype=np.int32),
    )
    (args.output_dir / "object_labels.json").write_text(
        json.dumps(labels, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "text_model": TEXT_MODEL,
        "frames": frame_count,
        "labels": len(labels),
        "object_records": int(len(frame_ids)),
        "videos": len(video_ids),
        "min_object_score": args.min_object_score,
    }
    (args.output_dir / "hybrid_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build object and metadata hybrid signals")
    parser.add_argument("--metadata", type=Path, default=Path("index/metadata.sqlite3"))
    parser.add_argument("--objects-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("index/hybrid"))
    parser.add_argument("--min-object-score", type=float, default=0.2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    build_index(parse_args())
