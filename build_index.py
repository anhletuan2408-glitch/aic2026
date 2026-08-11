from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an exact FAISS cosine index from the supplied AIC CLIP features."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("index"))
    return parser.parse_args()


def read_mapping(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    args = parse_args()
    feature_dir = args.data_dir / "clip-features-32"
    mapping_dir = args.data_dir / "map-keyframes"
    media_dir = args.data_dir / "media-info"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_paths = sorted(feature_dir.glob("*.npy"))
    if not feature_paths:
        raise FileNotFoundError(f"No .npy files found under {feature_dir}")

    matrices: list[np.ndarray] = []
    records: list[tuple[int, str, int, int, float, float, str, str]] = []
    global_id = 0

    for feature_path in feature_paths:
        video_id = feature_path.stem
        mapping_path = mapping_dir / f"{video_id}.csv"
        if not mapping_path.exists():
            raise FileNotFoundError(f"Missing mapping: {mapping_path}")

        matrix = np.load(feature_path, allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[1] != 512:
            raise ValueError(f"Unexpected feature shape for {video_id}: {matrix.shape}")

        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        faiss.normalize_L2(matrix)
        rows = read_mapping(mapping_path)
        if len(rows) != len(matrix):
            raise ValueError(
                f"Feature/map mismatch for {video_id}: {len(matrix)} != {len(rows)}"
            )

        media_path = media_dir / f"{video_id}.json"
        title = ""
        description = ""
        if media_path.exists():
            with media_path.open("r", encoding="utf-8") as stream:
                media = json.load(stream)
            title = str(media.get("title", ""))
            description = str(media.get("description", ""))

        for local_index, row in enumerate(rows):
            keyframe_no = int(row["n"])
            if keyframe_no != local_index + 1:
                raise ValueError(
                    f"Non-sequential keyframe number in {mapping_path}: {keyframe_no}"
                )
            records.append(
                (
                    global_id,
                    video_id,
                    keyframe_no,
                    int(row["frame_idx"]),
                    float(row["pts_time"]),
                    float(row["fps"]),
                    title,
                    description,
                )
            )
            global_id += 1
        matrices.append(matrix)

    vectors = np.concatenate(matrices, axis=0)
    if len(vectors) != len(records):
        raise RuntimeError("Internal vector/metadata alignment error")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(output_dir / "keyframes.faiss"))

    database_path = output_dir / "metadata.sqlite3"
    if database_path.exists():
        database_path.unlink()
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            """
            CREATE TABLE keyframes (
                global_id INTEGER PRIMARY KEY,
                video_id TEXT NOT NULL,
                keyframe_no INTEGER NOT NULL,
                frame_idx INTEGER NOT NULL,
                pts_time REAL NOT NULL,
                fps REAL NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO keyframes VALUES (?, ?, ?, ?, ?, ?, ?, ?)", records
        )
        connection.execute("CREATE INDEX idx_keyframes_video ON keyframes(video_id)")
        connection.commit()
    finally:
        connection.close()

    manifest = {
        "model": "ViT-B-32-quickgelu",
        "pretrained": "openai",
        "dimension": 512,
        "metric": "cosine_via_inner_product",
        "vectors": int(index.ntotal),
        "videos": len(feature_paths),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
