from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the AIC keyframe index.")
    parser.add_argument("query", help="Original query text")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Extra query wording/English translation; repeat as needed",
    )
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--candidate-k", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--per-video", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def encode_queries(texts: list[str], device: str) -> np.ndarray:
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    tokens = tokenizer(texts).to(device)
    model.eval()
    with torch.inference_mode():
        features = model.encode_text(tokens).float()
        features /= features.norm(dim=-1, keepdim=True)
        combined = features.mean(dim=0, keepdim=True)
        combined /= combined.norm(dim=-1, keepdim=True)
    return np.ascontiguousarray(combined.cpu().numpy(), dtype=np.float32)


def load_metadata(
    database_path: Path, global_ids: list[int]
) -> dict[int, sqlite3.Row]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        result: dict[int, sqlite3.Row] = {}
        chunk_size = 900
        for offset in range(0, len(global_ids), chunk_size):
            chunk = global_ids[offset : offset + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT * FROM keyframes WHERE global_id IN ({placeholders})"
            for row in connection.execute(query, chunk):
                result[int(row["global_id"])] = row
        return result
    finally:
        connection.close()


def main() -> None:
    args = parse_args()
    texts = [args.query, *args.variant]
    device = choose_device(args.device)
    query_vector = encode_queries(texts, device)

    index = faiss.read_index(str(args.index_dir / "keyframes.faiss"))
    candidate_k = min(args.candidate_k, index.ntotal)
    scores, ids = index.search(query_vector, candidate_k)
    ranked_ids = [int(value) for value in ids[0] if value >= 0]
    metadata = load_metadata(args.index_dir / "metadata.sqlite3", ranked_ids)

    selected: list[dict[str, object]] = []
    per_video: dict[str, int] = {}
    for rank, (global_id, score) in enumerate(zip(ranked_ids, scores[0]), start=1):
        row = metadata[global_id]
        video_id = str(row["video_id"])
        if per_video.get(video_id, 0) >= args.per_video:
            continue
        per_video[video_id] = per_video.get(video_id, 0) + 1
        selected.append(
            {
                "rank": len(selected) + 1,
                "retrieval_rank": rank,
                "video_id": video_id,
                "keyframe_no": int(row["keyframe_no"]),
                "frame_idx": int(row["frame_idx"]),
                "pts_time": float(row["pts_time"]),
                "score": float(score),
                "title": str(row["title"]),
            }
        )
        if len(selected) >= args.top_k:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0].keys()))
        writer.writeheader()
        writer.writerows(selected)

    submission_path = args.output.with_name(args.output.stem + "_submission.csv")
    with submission_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for row in selected:
            writer.writerow([row["video_id"], row["frame_idx"]])

    print(f"Device: {device}")
    print(f"Queries: {texts}")
    print(f"Wrote {len(selected)} ranked rows to {args.output}")
    print(f"Wrote KIS submission rows to {submission_path}")


if __name__ == "__main__":
    main()
