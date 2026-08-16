from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from benchmark_candidates import score_candidate_rows
from evaluate_suite import load_ground_truth
from submission import TOP_THRESHOLDS


def call_search_api(
    base_url: str,
    query: str,
    quality: bool,
    use_ocr: bool,
    use_hybrid: bool,
) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/search",
        data=json.dumps({
            "query": query,
            "top_k": 100,
            "candidate_k": 10000,
            "per_video": 3,
            "min_time_gap": 1.5,
            "quality": quality,
            "use_ocr": use_ocr,
            "use_hybrid": use_hybrid,
        }, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return list(json.loads(response.read())["results"])


def run_benchmark(
    ground_truth: Path,
    base_url: str,
    quality: bool,
    use_ocr: bool,
    use_hybrid: bool,
) -> dict[str, Any]:
    records = [
        record for record in load_ground_truth(ground_truth)
        if record["task"] == "kis"
    ]
    if not records:
        raise ValueError("Ground truth contains no KIS records")
    started = time.perf_counter()
    details = []
    for index, record in enumerate(records, start=1):
        before = time.perf_counter()
        rows = call_search_api(
            base_url, str(record["query"]), quality, use_ocr, use_hybrid
        )
        detail = {
            "query_id": record["query_id"],
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
        "config": {
            "quality": quality,
            "use_ocr": use_ocr,
            "use_hybrid": use_hybrid,
        },
        "metrics": metrics,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablate the live KIS retrieval signals against frame GT"
    )
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument(
        "--quality", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--ocr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--hybrid", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(
        args.ground_truth, args.base_url, args.quality, args.ocr, args.hybrid
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
