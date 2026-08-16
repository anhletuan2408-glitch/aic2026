from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

from diagnose_signal_ranks import first_target_rank, target_global_ids
from evaluate_suite import load_ground_truth
from hybrid_search import HybridConfig, HybridSignals
from ocr_index import OCRSignals
from retrieval_enhancements import qa_answer_hypothesis_queries, qa_retrieval_query


def round_robin_rankings(rankings: list[list[int]], limit: int = 100) -> list[int]:
    """Interleave signals/variants so one long noisy list cannot consume Top-100."""
    output: list[int] = []
    seen: set[int] = set()
    depth = 0
    while len(output) < limit and any(depth < len(items) for items in rankings):
        for items in rankings:
            if depth >= len(items):
                continue
            global_id = int(items[depth])
            if global_id not in seen:
                seen.add(global_id)
                output.append(global_id)
                if len(output) >= limit:
                    break
        depth += 1
    return output


def ocr_target_coverage(path: Path, targets: set[int]) -> int:
    if not targets or not path.exists():
        return 0
    placeholders = ",".join("?" for _ in targets)
    with closing(sqlite3.connect(path, timeout=10.0)) as connection:
        row = connection.execute(
            f"SELECT COUNT(*) FROM ocr_frames WHERE global_id IN ({placeholders})",
            tuple(sorted(targets)),
        ).fetchone()
    return int(row[0])


def diagnose(
    ground_truth: Path, index_dir: Path, object_depth: int = 2500,
    signal_limit: int = 100,
) -> dict[str, Any]:
    records = [
        item for item in load_ground_truth(ground_truth) if item["task"] == "qa"
    ]
    if not records:
        raise ValueError("Ground truth contains no QA records")
    hybrid = HybridSignals(index_dir / "hybrid", device="cpu")
    ocr_path = index_dir / "ocr.sqlite3"
    ocr = OCRSignals(ocr_path)
    config = replace(
        HybridConfig(), object_candidates=min(object_depth, hybrid.frame_count)
    )
    details: list[dict[str, Any]] = []
    for ordinal, record in enumerate(records, start=1):
        question = str(record["query"])
        scene = qa_retrieval_query(question)
        variants = list(dict.fromkeys([
            scene, question, *qa_answer_hypothesis_queries(question)
        ]))
        vectors = hybrid.encode_many(variants)
        object_rankings: list[list[int]] = []
        ocr_rankings: list[list[int]] = []
        metadata_ranks: list[int | None] = []
        labels: list[list[str]] = []
        for query, vector in zip(variants, vectors):
            object_ids, matched = hybrid.object_ranking(vector, config)
            object_rankings.append(object_ids)
            ocr_rankings.append(ocr.ranking(query, limit=object_depth))
            metadata_ranks.append(
                hybrid.metadata_video_ranks(vector).get(str(record["video_id"]))
            )
            labels.append([name for name, _ in matched[:5]])
        targets = target_global_ids(
            index_dir / "metadata.sqlite3", str(record["video_id"]),
            int(record["start"]), int(record["end"]),
        )
        object_fair = round_robin_rankings(object_rankings, signal_limit)
        ocr_fair = round_robin_rankings(ocr_rankings, signal_limit)
        combined_fair = round_robin_rankings(
            [*object_rankings, *ocr_rankings], signal_limit
        )
        object_evidence = first_target_rank(object_fair, targets)
        ocr_evidence = first_target_rank(ocr_fair, targets)
        combined_evidence = first_target_rank(combined_fair, targets)
        detail = {
            "query_id": record["query_id"],
            "target": {
                "video_id": record["video_id"],
                "start": record["start"], "end": record["end"],
                "ids": sorted(targets),
                "ocr_indexed": ocr_target_coverage(ocr_path, targets),
            },
            "variants": variants,
            "top_object_labels": labels,
            "metadata_best_video_rank": min(
                (rank for rank in metadata_ranks if rank is not None),
                default=None,
            ),
            "object_round_robin_rank": object_evidence.rank,
            "ocr_round_robin_rank": ocr_evidence.rank,
            "combined_round_robin_rank": combined_evidence.rank,
        }
        details.append(detail)
        print(
            f"[{ordinal}/{len(records)}] {record['query_id']} "
            f"object={object_evidence.rank} ocr={ocr_evidence.rank} "
            f"combined={combined_evidence.rank}", flush=True,
        )
    return {
        "queries": len(details),
        "ocr_frames": ocr.count(),
        "object_depth": object_depth,
        "signal_limit": signal_limit,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose QA frame recall in OCR/object/metadata signals"
    )
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--object-depth", type=int, default=2500)
    parser.add_argument("--signal-limit", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(
        args.ground_truth, args.index_dir, args.object_depth, args.signal_limit
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
