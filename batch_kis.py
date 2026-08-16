from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch

from search import choose_device, load_metadata
from search_kis import bounded_positive_int, select_candidates
from submission import MAX_ANSWERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run many Textual KIS queries with one CLIP model load."
    )
    parser.add_argument("queries", type=Path, help="JSONL query file")
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/batch"))
    parser.add_argument("--limit", type=bounded_positive_int)
    parser.add_argument("--encode-batch-size", type=bounded_positive_int, default=64)
    parser.add_argument("--candidate-k", type=bounded_positive_int, default=5000)
    parser.add_argument(
        "--top-k",
        type=lambda value: bounded_positive_int(value, MAX_ANSWERS),
        default=MAX_ANSWERS,
    )
    parser.add_argument("--per-video", type=bounded_positive_int, default=3)
    parser.add_argument("--min-time-gap", type=float, default=2.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_queries(path: Path, limit: int | None = None) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            query_id = str(payload.get("query_id", "")).strip()
            query = str(payload.get("query", "")).strip()
            variants = [str(value).strip() for value in payload.get("variants", [])]
            if not query_id or not query or any(not value for value in variants):
                raise ValueError(f"Invalid query at line {line_number}")
            if query_id in seen_ids:
                raise ValueError(f"Duplicate query_id: {query_id}")
            seen_ids.add(query_id)
            records.append(
                {"query_id": query_id, "texts": [query, *variants], "query": query}
            )
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError("Query file contains no usable records")
    return records


def encode_query_groups(
    records: list[dict[str, object]], device: str, batch_size: int
) -> np.ndarray:
    flat_texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for record in records:
        start = len(flat_texts)
        flat_texts.extend(str(value) for value in record["texts"])
        spans.append((start, len(flat_texts)))

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    model.eval()
    encoded_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(flat_texts), batch_size):
            tokens = tokenizer(flat_texts[offset : offset + batch_size]).to(device)
            features = model.encode_text(tokens).float()
            features /= features.norm(dim=-1, keepdim=True)
            encoded_batches.append(features.cpu().numpy())
    encoded = np.concatenate(encoded_batches, axis=0)

    grouped: list[np.ndarray] = []
    for start, end in spans:
        vector = encoded[start:end].mean(axis=0)
        vector /= np.linalg.norm(vector)
        grouped.append(vector)
    return np.ascontiguousarray(np.stack(grouped), dtype=np.float32)


def write_outputs(
    output_dir: Path,
    records: list[dict[str, object]],
    selected_by_query: list[list[dict[str, object]]],
    timings: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked_path = output_dir / "ranked.csv"
    with ranked_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = [
            "query_id",
            "query",
            "rank",
            "retrieval_rank",
            "video_id",
            "keyframe_no",
            "frame_idx",
            "pts_time",
            "score",
            "title",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record, selected in zip(records, selected_by_query):
            for row in selected:
                writer.writerow(
                    {
                        "query_id": record["query_id"],
                        "query": record["query"],
                        **row,
                    }
                )

    submission_path = output_dir / "submission.csv"
    with submission_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "rank", "video_id", "frame_id"])
        for record, selected in zip(records, selected_by_query):
            for row in selected:
                writer.writerow(
                    [
                        record["query_id"],
                        row["rank"],
                        row["video_id"],
                        row["frame_idx"],
                    ]
                )

    summary = {
        "queries": len(records),
        "answers": sum(len(rows) for rows in selected_by_query),
        "answers_per_query": [len(rows) for rows in selected_by_query],
        "timings_seconds": timings,
        "queries_per_second": len(records) / timings["total"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    records = load_queries(args.queries, args.limit)
    device = choose_device(args.device)

    encode_started = time.perf_counter()
    query_vectors = encode_query_groups(records, device, args.encode_batch_size)
    encode_seconds = time.perf_counter() - encode_started

    index = faiss.read_index(str(args.index_dir / "keyframes.faiss"))
    candidate_k = min(args.candidate_k, index.ntotal)
    search_started = time.perf_counter()
    scores, ids = index.search(query_vectors, candidate_k)
    search_seconds = time.perf_counter() - search_started

    all_ids = sorted({int(value) for value in ids.flat if value >= 0})
    metadata = load_metadata(args.index_dir / "metadata.sqlite3", all_ids)
    selected_by_query: list[list[dict[str, object]]] = []
    for query_ids, query_scores in zip(ids, scores):
        ranked_ids = [int(value) for value in query_ids if value >= 0]
        selected = select_candidates(
            ranked_ids,
            query_scores,
            metadata,
            args.top_k,
            args.per_video,
            args.min_time_gap,
        )
        if len(selected) != args.top_k:
            raise RuntimeError(
                f"Expected {args.top_k} answers, received {len(selected)}"
            )
        selected_by_query.append(selected)

    total_seconds = time.perf_counter() - started
    timings = {
        "encode": encode_seconds,
        "faiss_search": search_seconds,
        "total": total_seconds,
    }
    write_outputs(args.output_dir, records, selected_by_query, timings)
    print(f"Device: {device}")
    print(f"Queries: {len(records)}")
    print(f"Answers: {sum(len(rows) for rows in selected_by_query)}")
    print(f"Encode seconds: {encode_seconds:.3f}")
    print(f"FAISS search seconds: {search_seconds:.3f}")
    print(f"Total seconds: {total_seconds:.3f}")
    print(f"Queries/second: {len(records) / total_seconds:.3f}")


if __name__ == "__main__":
    main()
