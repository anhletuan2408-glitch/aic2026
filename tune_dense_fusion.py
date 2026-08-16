from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retrieval_enhancements import expand_query, fuse_query_rankings


THRESHOLDS = (1, 5, 20, 50, 100)


@dataclass(frozen=True)
class DenseCase:
    query_id: str
    task: str
    target_ids: set[int]
    clip_ids: list[int]
    siglip_ids: list[int]


def first_target_rank(ranking: list[int], targets: set[int]) -> int | None:
    return next(
        (rank for rank, global_id in enumerate(ranking, start=1)
         if global_id in targets),
        None,
    )


def official_score(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return sum(rank <= threshold for threshold in THRESHOLDS) / len(THRESHOLDS)


def fused_rank(case: DenseCase, siglip_weight: float,
               limit: int) -> int | None:
    if siglip_weight == 0.0:
        ranking = case.clip_ids[:limit]
    else:
        ranking, _ = fuse_query_rankings(
            [case.clip_ids, case.siglip_ids],
            weights=[1.0, siglip_weight],
            limit=limit,
        )
    return first_target_rank(ranking, case.target_ids)


def summarize(cases: list[DenseCase], weights: list[float],
              limit: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for weight in weights:
        ranks = [fused_rank(case, weight, limit) for case in cases]
        task_scores: dict[str, list[float]] = {}
        for case, rank in zip(cases, ranks):
            task_scores.setdefault(case.task, []).append(official_score(rank))
        output[str(weight)] = {
            "final_score": sum(map(official_score, ranks)) / max(1, len(ranks)),
            "tasks": {
                task: sum(scores) / len(scores)
                for task, scores in task_scores.items()
            },
            "ranks": {
                case.query_id: rank for case, rank in zip(cases, ranks)
            },
        }
    return output


def collect_cases(engine: Any,
                  records: list[dict[str, Any]],
                  candidate_k: int) -> list[DenseCase]:
    from diagnose_signal_ranks import target_global_ids

    if engine.siglip2_index is None or engine.reranker is None:
        raise RuntimeError("A completed global SigLIP2 index is required")
    cases: list[DenseCase] = []
    candidate_k = min(candidate_k, engine.index.ntotal)
    for index, record in enumerate(records, start=1):
        variants = expand_query(str(record["query"]))
        vectors = engine.encode_many(variants)
        _, clip_values = engine.index.search(vectors, candidate_k)
        clip_rankings = [
            [int(value) for value in row if value >= 0]
            for row in clip_values
        ]
        clip_ids, _ = fuse_query_rankings(clip_rankings, limit=candidate_k)
        siglip_vector = engine.reranker.encode_text(str(record["query"]))
        _, siglip_values = engine.siglip2_index.search(siglip_vector, candidate_k)
        siglip_ids = [int(value) for value in siglip_values[0] if value >= 0]
        targets = target_global_ids(
            engine.metadata_path,
            str(record["video_id"]),
            int(record["start"]),
            int(record["end"]),
        )
        cases.append(DenseCase(
            str(record["query_id"]), str(record["task"]), targets,
            clip_ids, siglip_ids,
        ))
        print(f"[{index}/{len(records)}] {record['query_id']}", flush=True)
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune CLIP/SigLIP2 RRF weights on local KIS/QA frame GT"
    )
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--zip-dir", type=Path, default=Path("E:/"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--candidate-k", type=int, default=10000)
    parser.add_argument(
        "--weights", default="0,0.25,0.5,0.75,1,1.5,2",
        help="Comma-separated SigLIP2 RRF weights",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from evaluate_suite import load_ground_truth
    from rerank import SIGLIP2_MODEL, Siglip2Reranker
    from web_app import KeyframeStore
    from web_app_vi import MultilingualFaissEngine
    weights = [float(value) for value in args.weights.split(",")]
    if any(weight < 0 for weight in weights):
        raise ValueError("Fusion weights must be non-negative")
    records = [
        record for record in load_ground_truth(args.ground_truth)
        if record["task"] in {"kis", "qa"}
    ]
    if not records:
        raise ValueError("Ground truth contains no KIS/QA frame records")
    store = KeyframeStore(args.zip_dir)
    try:
        reranker = Siglip2Reranker(
            store, args.device, model_name=SIGLIP2_MODEL
        )
        engine = MultilingualFaissEngine(
            args.index_dir, args.device, reranker=reranker,
            query_ensemble=True,
        )
        cases = collect_cases(engine, records, args.candidate_k)
        report = {
            "queries": len(cases),
            "candidate_k": args.candidate_k,
            "weights": summarize(cases, weights, args.candidate_k),
            "signals": {
                case.query_id: {
                    "clip_rank": first_target_rank(case.clip_ids, case.target_ids),
                    "siglip2_rank": first_target_rank(case.siglip_ids, case.target_ids),
                }
                for case in cases
            },
        }
    finally:
        store.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
