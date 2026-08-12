from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate import evaluate_kis, evaluate_qna, evaluate_trake, load_rows
from submission import TOP_THRESHOLDS, final_score

TASKS = {"kis", "qa", "trake"}


def load_ground_truth(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"query_id", "task", "query", "video_id"}
            if not required.issubset(record):
                raise ValueError(f"Missing fields at line {line_number}: {sorted(required - set(record))}")
            query_id = str(record["query_id"])
            task = str(record["task"]).casefold()
            if query_id in seen:
                raise ValueError(f"Duplicate query_id: {query_id}")
            if task not in TASKS:
                raise ValueError(f"Invalid task for {query_id}: {task}")
            if task in {"kis", "qa"} and not {"start", "end"}.issubset(record):
                raise ValueError(f"{task} truth needs start and end: {query_id}")
            if task == "qa" and not record.get("answers"):
                raise ValueError(f"QA truth needs accepted answers: {query_id}")
            if task == "trake" and not record.get("moments"):
                raise ValueError(f"TRAKE truth needs moments: {query_id}")
            record["query_id"], record["task"] = query_id, task
            records.append(record)
            seen.add(query_id)
    if not records:
        raise ValueError("Ground-truth JSONL is empty")
    return records


def score_record(record: dict[str, Any], prediction_dir: Path) -> dict[str, Any]:
    prediction = prediction_dir / f"{record['query_id']}.csv"
    if prediction.exists():
        rows = load_rows(prediction)
        evaluator = {"kis": evaluate_kis, "qa": evaluate_qna, "trake": evaluate_trake}[record["task"]]
        scores = evaluator(rows, record)
    else:
        scores = [0.0] * max(TOP_THRESHOLDS)
    top, score = final_score(scores)
    return {
        "query_id": record["query_id"],
        "task": record["task"],
        "query": record["query"],
        "prediction": str(prediction),
        "missing": not prediction.exists(),
        **{f"r@{k}": top[k] for k in TOP_THRESHOLDS},
        "final_score": score,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics = [f"r@{k}" for k in TOP_THRESHOLDS] + ["final_score"]
    return {name: sum(float(row[name]) for row in rows) / len(rows) for name in metrics}


def evaluate_suite(ground_truth: Path, prediction_dir: Path) -> dict[str, Any]:
    records = load_ground_truth(ground_truth)
    details = [score_record(record, prediction_dir) for record in records]
    by_task = {
        task: aggregate([row for row in details if row["task"] == task])
        for task in sorted(TASKS)
        if any(row["task"] == task for row in details)
    }
    return {
        "queries": len(details),
        "missing_predictions": sum(bool(row["missing"]) for row in details),
        "overall": aggregate(details),
        "tasks": by_task,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KIS, QA and TRAKE with official Top-k R-Scores")
    parser.add_argument("ground_truth", type=Path, help="UTF-8 JSONL ground truth")
    parser.add_argument("prediction_dir", type=Path, help="Directory containing <query_id>.csv")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark_suite.json"))
    args = parser.parse_args()
    report = evaluate_suite(args.ground_truth, args.prediction_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()