from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Mapping

import faiss
import numpy as np

from search import choose_device, encode_queries, load_metadata
from submission import KISAnswer, MAX_ANSWERS, write_kis_submission


def bounded_positive_int(value: str, maximum: int | None = None) -> int:
    parsed = int(value)
    if parsed < 1 or (maximum is not None and parsed > maximum):
        suffix = f" and at most {maximum}" if maximum is not None else ""
        raise argparse.ArgumentTypeError(f"value must be positive{suffix}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Textual KIS retrieval with the supplied FAISS index."
    )
    parser.add_argument("query", help="Original query text")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Extra wording or English translation; repeat as needed",
    )
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--candidate-k", type=bounded_positive_int, default=5000)
    parser.add_argument(
        "--top-k",
        type=lambda value: bounded_positive_int(value, MAX_ANSWERS),
        default=MAX_ANSWERS,
    )
    parser.add_argument("--per-video", type=bounded_positive_int, default=3)
    parser.add_argument(
        "--min-time-gap",
        type=float,
        default=2.0,
        help="Minimum seconds between accepted frames from the same video",
    )
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def select_candidates(
    ranked_ids: list[int],
    scores: np.ndarray,
    metadata: Mapping[int, Mapping[str, object]],
    top_k: int,
    per_video_limit: int,
    min_time_gap: float,
    video_pool_limit: int | None = None,
) -> list[dict[str, object]]:
    if min_time_gap < 0:
        raise ValueError("min_time_gap must be non-negative")

    grouped: dict[str, list[dict[str, object]]] = {}
    video_order: list[str] = []
    for retrieval_rank, (global_id, score) in enumerate(
        zip(ranked_ids, scores), start=1
    ):
        row = metadata[global_id]
        video_id = str(row["video_id"])
        candidates = grouped.setdefault(video_id, [])
        if not candidates:
            video_order.append(video_id)
        if len(candidates) >= per_video_limit:
            continue
        pts_time = float(row["pts_time"])
        if any(
            abs(pts_time - float(existing["pts_time"])) < min_time_gap
            for existing in candidates
        ):
            continue
        candidates.append(
            {
                "_global_id": global_id,
                "retrieval_rank": retrieval_rank,
                "video_id": video_id,
                "keyframe_no": int(row["keyframe_no"]),
                "frame_idx": int(row["frame_idx"]),
                "pts_time": pts_time,
                "score": float(score),
                "title": str(row["title"]),
            }
        )

    if video_pool_limit is not None:
        video_order = video_order[:video_pool_limit]
    selected: list[dict[str, object]] = []
    for round_index in range(per_video_limit):
        for video_id in video_order:
            candidates = grouped[video_id]
            if round_index >= len(candidates):
                continue
            selected.append({"rank": len(selected) + 1, **candidates[round_index]})
            if len(selected) >= top_k:
                return selected
    return selected

def protect_signal_rows(
    rows: list[dict[str, object]], max_protected: int = 4,
    max_ocr_rank: int = 10,
) -> list[dict[str, object]]:
    """Keep the visual winner, then expose a bounded exact-text prefix."""
    if not rows or max_protected <= 0:
        return rows
    first = rows[0]
    first_key = (str(first["video_id"]), int(first["frame_idx"]))
    protected: list[dict[str, object]] = []
    protected_videos: set[str] = set()
    for row in sorted(
        rows[1:],
        key=lambda item: int(item.get("_ocr_rank", max_ocr_rank + 1)),
    ):
        ocr_rank = int(row.get("_ocr_rank", max_ocr_rank + 1))
        video_id = str(row["video_id"])
        key = (video_id, int(row["frame_idx"]))
        if ocr_rank > max_ocr_rank or key == first_key or video_id in protected_videos:
            continue
        protected.append(row)
        protected_videos.add(video_id)
        if len(protected) >= max_protected:
            break
    protected_keys = {
        (str(row["video_id"]), int(row["frame_idx"])) for row in protected
    }
    output = [first, *protected]
    output.extend(
        row for row in rows[1:]
        if (str(row["video_id"]), int(row["frame_idx"])) not in protected_keys
    )
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def diversify_ranked_rows(
    rows: list[dict[str, object]], top_k: int,
    unique_prefix: int = 20, per_video_limit: int = 3,
) -> list[dict[str, object]]:
    """Protect early-rank video coverage, then admit alternate frames."""
    chosen: list[dict[str, object]] = []
    seen_rows: set[tuple[str, int]] = set()
    counts: dict[str, int] = {}
    for row in rows:
        video_id = str(row["video_id"])
        key = (video_id, int(row["frame_idx"]))
        if video_id in counts or key in seen_rows:
            continue
        chosen.append(row)
        seen_rows.add(key)
        counts[video_id] = 1
        if len(chosen) >= min(unique_prefix, top_k):
            break
    for row in rows:
        video_id = str(row["video_id"])
        key = (video_id, int(row["frame_idx"]))
        if key in seen_rows or counts.get(video_id, 0) >= per_video_limit:
            continue
        chosen.append(row)
        seen_rows.add(key)
        counts[video_id] = counts.get(video_id, 0) + 1
        if len(chosen) >= top_k:
            break
    for rank, row in enumerate(chosen, start=1):
        row["rank"] = rank
    return chosen

def main() -> None:
    args = parse_args()
    texts = [args.query, *args.variant]
    if any(not text.strip() for text in texts):
        raise ValueError("Query text and variants must not be blank")

    device = choose_device(args.device)
    query_vector = encode_queries(texts, device)
    index = faiss.read_index(str(args.index_dir / "keyframes.faiss"))
    candidate_k = min(args.candidate_k, index.ntotal)
    scores, ids = index.search(query_vector, candidate_k)
    ranked_ids = [int(value) for value in ids[0] if value >= 0]
    metadata = load_metadata(args.index_dir / "metadata.sqlite3", ranked_ids)
    selected = select_candidates(
        ranked_ids,
        scores[0],
        metadata,
        args.top_k,
        args.per_video,
        args.min_time_gap,
    )
    if not selected:
        raise RuntimeError("FAISS search returned no candidates")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0].keys()))
        writer.writeheader()
        writer.writerows(selected)

    submission_path = args.output.with_name(args.output.stem + "_submission.csv")
    write_kis_submission(
        submission_path,
        [
            KISAnswer(str(row["video_id"]), int(row["frame_idx"]))
            for row in selected
        ],
    )
    print(f"Device: {device}")
    print(f"FAISS candidates: {candidate_k}")
    print(f"Wrote {len(selected)} ranked rows to {args.output}")
    print(f"Wrote KIS submission rows to {submission_path}")


if __name__ == "__main__":
    main()
