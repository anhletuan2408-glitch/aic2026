from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from hybrid_search import HybridConfig, HybridSignals, reciprocal_rank_fusion
from retrieval_enhancements import (
    expand_query, fuse_query_rankings, temporal_neighbor_ranking,
)
from web_app_vi import MODEL_NAME

THRESHOLDS = (1, 5, 20, 50, 100)


def first_video_ranks(rankings, video_by_frame, target_videos):
    found = []
    for ids, target in zip(rankings, target_videos):
        rank = 1000000
        seen = set()
        for global_id in ids:
            video_id = str(video_by_frame[int(global_id)])
            if video_id in seen:
                continue
            seen.add(video_id)
            if video_id == target:
                rank = len(seen)
                break
        found.append(rank)
    return found


def recall(ranks):
    return {f"hit@{k}": sum(rank <= k for rank in ranks) / len(ranks) for k in THRESHOLDS}


def load_video_data(metadata, limit):
    with sqlite3.connect(metadata) as connection:
        rows = connection.execute(
            "SELECT video_id, title FROM keyframes GROUP BY video_id "
            "ORDER BY MIN(global_id) LIMIT ?", (limit,)
        ).fetchall()
        frame_rows = connection.execute(
            "SELECT global_id, video_id FROM keyframes ORDER BY global_id"
        ).fetchall()
    return (
        [str(title).strip() for _, title in rows],
        [str(video_id) for video_id, _ in rows],
        np.asarray([str(video) for _, video in frame_rows]),
    )


def main():
    parser = argparse.ArgumentParser(description="A/B proxy benchmark for hybrid retrieval")
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--candidate-k", type=int, default=5000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, default=Path("outputs/hybrid_benchmark.json"))
    args = parser.parse_args()

    started = time.perf_counter()
    queries, targets, video_by_frame = load_video_data(
        args.index_dir / "metadata.sqlite3", args.limit
    )
    visual_model = SentenceTransformer(MODEL_NAME, device=args.device)
    query_vectors = visual_model.encode(
        queries, batch_size=64, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    index = faiss.read_index(str(args.index_dir / "keyframes.faiss"))
    _, base_ids = index.search(query_vectors, min(args.candidate_k, index.ntotal))
    base_ranks = first_video_ranks(base_ids, video_by_frame, targets)
    expanded = [expand_query(query) for query in queries]
    flat_queries = [variant for variants in expanded for variant in variants]
    flat_vectors = visual_model.encode(
        flat_queries, batch_size=64, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    _, flat_ids = index.search(flat_vectors, min(args.candidate_k, index.ntotal))
    ensemble_rankings = []
    offset = 0
    for variants in expanded:
        count = len(variants)
        ranking, _ = fuse_query_rankings([
            [int(value) for value in row if value >= 0]
            for row in flat_ids[offset : offset + count]
        ], limit=args.candidate_k)
        ensemble_rankings.append(ranking)
        offset += count

    signals = HybridSignals(args.index_dir / "hybrid", device="cpu")
    text_vectors = signals.encode_many(queries)
    config = HybridConfig()
    hybrid_rankings = []
    ensemble_only_rankings = []
    temporal_only_rankings = []
    enhanced_rankings = []
    video_map = {index: str(video) for index, video in enumerate(video_by_frame)}
    for query_vector, ids, ensemble in zip(text_vectors, base_ids, ensemble_rankings):
        base = [int(value) for value in ids if value >= 0]
        objects, _ = signals.object_ranking(query_vector, config)
        union = set(base) | set(objects)
        videos = {value: str(video_by_frame[value]) for value in union}
        fused, _ = reciprocal_rank_fusion(
            base, objects, videos, signals.metadata_video_ranks(query_vector), config
        )
        hybrid_rankings.append(fused)
        ensemble_only, _ = reciprocal_rank_fusion(
            ensemble, objects, video_map,
            signals.metadata_video_ranks(query_vector), config,
        )
        ensemble_only_rankings.append(ensemble_only)
        base_temporal = temporal_neighbor_ranking(
            base, video_map, len(video_by_frame)
        )
        temporal_only, _ = reciprocal_rank_fusion(
            base, objects, video_map,
            signals.metadata_video_ranks(query_vector), config,
            temporal_ids=base_temporal,
        )
        temporal_only_rankings.append(temporal_only)
        temporal = temporal_neighbor_ranking(
            ensemble, video_map, len(video_by_frame)
        )
        enhanced, _ = reciprocal_rank_fusion(
            ensemble, objects, video_map,
            signals.metadata_video_ranks(query_vector), config,
            temporal_ids=temporal,
        )
        enhanced_rankings.append(enhanced)
    hybrid_ranks = first_video_ranks(hybrid_rankings, video_by_frame, targets)
    ensemble_only_ranks = first_video_ranks(
        ensemble_only_rankings, video_by_frame, targets
    )
    temporal_only_ranks = first_video_ranks(
        temporal_only_rankings, video_by_frame, targets
    )
    enhanced_ranks = first_video_ranks(enhanced_rankings, video_by_frame, targets)
    report = {
        "kind": "metadata-title proxy; not organizer accuracy",
        "queries": len(queries),
        "baseline": recall(base_ranks),
        "hybrid": recall(hybrid_ranks),
        "hybrid_ensemble": recall(ensemble_only_ranks),
        "hybrid_temporal": recall(temporal_only_ranks),
        "hybrid_enhanced": recall(enhanced_ranks),
        "seconds": round(time.perf_counter() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
