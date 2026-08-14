from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np

from hybrid_search import HybridConfig, reciprocal_rank_fusion
from retrieval_enhancements import expand_query, fuse_query_rankings
from search import load_metadata
from web_app_vi import MultilingualFaissEngine


@dataclass(frozen=True)
class RankEvidence:
    rank: int | None
    searched: int
    target_hits: int


def first_target_rank(
    ranking: Iterable[int], target_ids: set[int]
) -> RankEvidence:
    searched = 0
    hits = 0
    first: int | None = None
    for searched, global_id in enumerate(ranking, start=1):
        if int(global_id) in target_ids:
            hits += 1
            if first is None:
                first = searched
    return RankEvidence(first, searched, hits)


def target_global_ids(
    metadata_path: Path, video_id: str, start: int, end: int
) -> set[int]:
    if start < 0 or end < start:
        raise ValueError("Target frame range is invalid")
    with closing(sqlite3.connect(metadata_path)) as connection:
        rows = connection.execute(
            "SELECT global_id FROM keyframes "
            "WHERE video_id=? AND frame_idx BETWEEN ? AND ? ORDER BY global_id",
            (video_id, start, end),
        ).fetchall()
    ids = {int(row[0]) for row in rows}
    if not ids:
        raise ValueError(
            f"No indexed keyframe for {video_id} in frame range [{start}, {end}]"
        )
    return ids


def _rank_payload(
    ranking: Iterable[int], target_ids: set[int]
) -> dict[str, int | None]:
    return asdict(first_target_rank(ranking, target_ids))


def trace_query(
    engine: MultilingualFaissEngine,
    query: str,
    video_id: str,
    start: int,
    end: int,
    candidate_k: int = 5000,
) -> dict[str, Any]:
    target_ids = target_global_ids(engine.metadata_path, video_id, start, end)
    candidate_k = min(max(candidate_k, 100), engine.index.ntotal)

    variants = expand_query(query)
    vectors = engine.encode_many(variants)
    _, full_clip_ids = engine.index.search(vectors[:1], engine.index.ntotal)
    _, candidate_ids = engine.index.search(vectors, candidate_k)
    clip_rankings = [
        [int(value) for value in row if value >= 0] for row in candidate_ids
    ]
    expanded_ids, _ = fuse_query_rankings(clip_rankings, limit=candidate_k)
    base_ids = clip_rankings[0]

    signals: dict[str, dict[str, int | None]] = {
        "clip_exact": _rank_payload(full_clip_ids[0], target_ids),
        "clip_expanded": _rank_payload(expanded_ids, target_ids),
    }
    fused_ids = expanded_ids
    metadata_video_rank: int | None = None
    if engine.hybrid is not None:
        text_vector = engine.hybrid.encode(query)
        full_object_config = replace(
            engine.hybrid_config, object_candidates=engine.hybrid.frame_count
        )
        object_ids, _ = engine.hybrid.object_ranking(
            text_vector, full_object_config
        )
        current_object_ids = object_ids[: engine.hybrid_config.object_candidates]
        ocr_ids = (
            engine.ocr.ranking(query, limit=engine.index.ntotal)
            if engine.ocr is not None else []
        )
        current_ocr_ids = ocr_ids[:2500]
        union_ids = list(dict.fromkeys([
            *expanded_ids, *current_object_ids, *current_ocr_ids
        ]))
        metadata = load_metadata(engine.metadata_path, union_ids)
        video_by_id = {
            global_id: str(row["video_id"])
            for global_id, row in metadata.items()
        }
        metadata_ranks = engine.hybrid.metadata_video_ranks(text_vector)
        metadata_video_rank = metadata_ranks.get(video_id)
        no_metadata = replace(engine.hybrid_config, metadata_weight=0.0)
        clip_objects, _ = reciprocal_rank_fusion(
            expanded_ids,
            current_object_ids,
            video_by_id,
            {},
            no_metadata,
        )
        clip_ocr, _ = reciprocal_rank_fusion(
            expanded_ids,
            [],
            video_by_id,
            {},
            no_metadata,
            ocr_ids=current_ocr_ids,
        )
        clip_metadata, _ = reciprocal_rank_fusion(
            expanded_ids,
            [],
            video_by_id,
            metadata_ranks,
            engine.hybrid_config,
        )
        fused_ids, _ = reciprocal_rank_fusion(
            expanded_ids,
            current_object_ids,
            video_by_id,
            metadata_ranks,
            engine.hybrid_config,
            ocr_ids=current_ocr_ids,
        )
        signals["objects"] = _rank_payload(object_ids, target_ids)
        signals["ocr"] = _rank_payload(ocr_ids, target_ids)
        signals["clip_plus_objects"] = _rank_payload(
            clip_objects, target_ids
        )
        signals["clip_plus_ocr"] = _rank_payload(clip_ocr, target_ids)
        signals["clip_plus_metadata"] = _rank_payload(
            clip_metadata, target_ids
        )
        signals["fused_preselection"] = _rank_payload(fused_ids, target_ids)

    fast_rows = engine.search(
        query,
        100,
        candidate_k=candidate_k,
        per_video=3,
        min_time_gap=2.0,
        quality=False,
    )
    fast_target_rank: int | None = None
    for rank, row in enumerate(fast_rows, start=1):
        if (
            str(row["video_id"]) == video_id
            and start <= int(row["frame_idx"]) <= end
        ):
            fast_target_rank = rank
            break
    signals["final_fast"] = {
        "rank": fast_target_rank,
        "searched": len(fast_rows),
        "target_hits": sum(
            str(row["video_id"]) == video_id
            and start <= int(row["frame_idx"]) <= end
            for row in fast_rows
        ),
    }
    return {
        "query": query,
        "target": {
            "video_id": video_id,
            "start": start,
            "end": end,
            "indexed_keyframes": len(target_ids),
        },
        "variants": variants,
        "candidate_k": candidate_k,
        "metadata_video_rank": metadata_video_rank,
        "signals": signals,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace a known frame range through every retrieval signal"
    )
    parser.add_argument("query")
    parser.add_argument("video_id")
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--candidate-k", type=int, default=5000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = MultilingualFaissEngine(
        args.index_dir, args.device, query_ensemble=True
    )
    report = trace_query(
        engine,
        args.query,
        args.video_id,
        args.start,
        args.end,
        args.candidate_k,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
