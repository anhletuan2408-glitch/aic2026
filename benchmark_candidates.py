from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from evaluate_suite import load_ground_truth
from submission import TOP_THRESHOLDS


def score_candidate_rows(
    rows: list[dict[str, Any]], truth: dict[str, Any]
) -> dict[str, float | int]:
    relevant = [
        str(row["video_id"]) == str(truth["video_id"])
        and int(truth["start"]) <= int(row["frame_idx"]) <= int(truth["end"])
        for row in rows
    ]
    metrics: dict[str, float | int] = {
        f"r@{k}": float(any(relevant[:k])) for k in TOP_THRESHOLDS
    }
    metrics["final_score"] = sum(
        float(metrics[f"r@{k}"]) for k in TOP_THRESHOLDS
    ) / len(TOP_THRESHOLDS)
    metrics["first_relevant_rank"] = next(
        (rank for rank, hit in enumerate(relevant, start=1) if hit), 0
    )
    same_video = [
        (rank, abs(int(row["frame_idx"]) - int(truth["start"])))
        for rank, row in enumerate(rows, start=1)
        if str(row["video_id"]) == str(truth["video_id"])
    ]
    metrics["first_same_video_rank"] = same_video[0][0] if same_video else 0
    metrics["closest_frame_distance"] = (
        min(distance for _, distance in same_video) if same_video else -1
    )
    return metrics


def call_candidate_api(base_url: str, question: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/qa/candidates",
        data=json.dumps({"question": question}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.loads(response.read())
    return list(payload["results"])


def run_candidate_benchmark(
    ground_truth: Path, base_url: str, cache_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    records = [
        record for record in load_ground_truth(ground_truth)
        if record["task"] == "qa"
    ]
    if not records:
        raise ValueError("Ground truth contains no QA records")
    started = time.perf_counter()
    details = []
    for index, record in enumerate(records, start=1):
        before = time.perf_counter()
        cache_path = (
            cache_dir / f"{record['query_id']}.json" if cache_dir else None
        )
        if cache_path and cache_path.exists() and not force:
            rows = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            rows = call_candidate_api(base_url, str(record["query"]))
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        detail = {
            "query_id": record["query_id"],
            "query": record["query"],
            "truth_video_id": record["video_id"],
            "truth_start": record["start"],
            "truth_end": record["end"],
            "candidate_count": len(rows),
            "elapsed_ms": round((time.perf_counter() - before) * 1000),
            **score_candidate_rows(rows, record),
        }
        details.append(detail)
        print(
            f"[{index}/{len(records)}] {record['query_id']} "
            f"rank={detail['first_relevant_rank']} "
            f"in {detail['elapsed_ms']}ms",
            flush=True,
        )
    metrics = {
        f"r@{k}": sum(float(row[f"r@{k}"]) for row in details) / len(details)
        for k in TOP_THRESHOLDS
    }
    metrics["final_score"] = sum(
        float(row["final_score"]) for row in details
    ) / len(details)
    return {
        "queries": len(details),
        "metrics": metrics,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure QA frame-candidate recall without running Qwen"
    )
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--report", type=Path,
        default=Path("outputs/qa_candidate_benchmark.json"),
    )
    args = parser.parse_args()
    report = run_candidate_benchmark(
        args.ground_truth, args.base_url, args.cache_dir, args.force
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
