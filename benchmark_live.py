from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from evaluate_suite import evaluate_suite, load_ground_truth


def wait_for_health(base_url: str, timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"{base_url.rstrip('/')}/api/health", timeout=5
            ) as response:
                health = json.loads(response.read())
            if health.get("status") == "ready":
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as error:
            last_error = error
        time.sleep(2)
    raise RuntimeError(f"API did not recover within {timeout:.0f}s") from last_error


def call_api(base_url: str, record: dict[str, Any], qa_candidates: int) -> dict[str, Any]:
    payload = {
        "task": record["task"],
        "query": record["query"],
        "top_k": 100,
        "candidate_k": 10000,
        "per_video": 3,
        "min_time_gap": 1.5,
        "quality": True,
        "qa_candidates": qa_candidates,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/assistant",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"API failed for {record['query_id']}: {detail}"
            ) from error
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            if attempt >= 3:
                raise
            print(
                f"API connection lost for {record['query_id']}; "
                f"waiting for restart ({attempt}/2)",
                flush=True,
            )
            wait_for_health(base_url)
    raise AssertionError("unreachable")

def result_rows(task: str, results: list[dict[str, Any]]) -> list[list[Any]]:
    if task == "kis":
        return [[row["video_id"], int(row["frame_idx"])] for row in results]
    if task == "qa":
        return [[row["video_id"], int(row["frame_idx"]), row["answer"]]
                for row in results]
    return [[row["video_id"], *[int(value) for value in row["frame_ids"]]]
            for row in results]


def write_prediction(path: Path, rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows[:100])
    temporary.replace(path)


def run_live(ground_truth: Path, output_dir: Path, base_url: str,
             qa_candidates: int, force: bool = False) -> dict[str, Any]:
    records = load_ground_truth(ground_truth)
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        path = output_dir / f"{record['query_id']}.csv"
        if path.exists() and not force:
            print(f"[{index}/{len(records)}] resume {record['query_id']}", flush=True)
            continue
        before = time.perf_counter()
        response = call_api(base_url, record, qa_candidates)
        write_prediction(path, result_rows(record["task"], response["results"]))
        print(f"[{index}/{len(records)}] {record['query_id']} {response['count']} rows "
              f"in {time.perf_counter()-before:.1f}s", flush=True)
    report = evaluate_suite(ground_truth, output_dir)
    report["wall_seconds"] = round(time.perf_counter() - started, 3)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and score GT queries against the live CUDA assistant")
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gt_predictions"))
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--qa-candidates", type=int, default=10, choices=(5, 8, 10))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("outputs/ground_truth_benchmark.json"))
    args = parser.parse_args()
    report = run_live(args.ground_truth, args.output_dir, args.base_url,
                      args.qa_candidates, args.force)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()