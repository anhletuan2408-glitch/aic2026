from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rerank import Siglip2Reranker
from submission import TOP_THRESHOLDS
from web_app import KeyframeStore
from web_app_vi import MultilingualFaissEngine


def load_validation(path: Path) -> list[dict[str, object]]:
    records = []
    seen = set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"query_id", "query", "video_id", "start", "end"}
            if not required.issubset(record):
                raise ValueError(f"Missing validation field at line {line_number}")
            query_id = str(record["query_id"])
            if query_id in seen:
                raise ValueError(f"Duplicate query_id: {query_id}")
            if int(record["start"]) > int(record["end"]):
                raise ValueError(f"Invalid frame range for {query_id}")
            seen.add(query_id)
            records.append(record)
    if not records:
        raise ValueError("Validation file is empty")
    return records


def score_ranking(rows: list[dict[str, object]], truth: dict[str, object]) -> dict[str, float]:
    relevant = [
        str(row["video_id"]) == str(truth["video_id"])
        and int(truth["start"]) <= int(row["frame_idx"]) <= int(truth["end"])
        for row in rows
    ]
    values = {
        f"r@{k}": float(any(relevant[: min(k, len(relevant))]))
        for k in TOP_THRESHOLDS
    }
    values["final_score"] = sum(values.values()) / len(TOP_THRESHOLDS)
    values["first_relevant_rank"] = float(
        next((index for index, hit in enumerate(relevant, start=1) if hit), 0)
    )
    return values


def aggregate(scored: list[dict[str, float]]) -> dict[str, float]:
    keys = [f"r@{k}" for k in TOP_THRESHOLDS] + ["final_score"]
    return {key: sum(row[key] for row in scored) / len(scored) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AIC retrieval modes on KIS truth")
    parser.add_argument("validation", type=Path)
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--zip-dir", type=Path, default=Path(".."))
    parser.add_argument("--mode", choices=["baseline", "hybrid", "quality"], default="quality")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, default=Path("outputs/validation.json"))
    args = parser.parse_args()

    records = load_validation(args.validation)
    keyframes = KeyframeStore(args.zip_dir) if args.mode == "quality" else None
    reranker = Siglip2Reranker(keyframes, args.device) if keyframes else None
    hybrid_dir = args.index_dir / "hybrid"
    if args.mode == "baseline":
        hybrid_dir = args.index_dir / "hybrid-disabled"
    engine = MultilingualFaissEngine(
        args.index_dir, args.device, hybrid_dir=hybrid_dir, reranker=reranker
    )
    started = time.perf_counter()
    details = []
    for record in records:
        rows = engine.search(str(record["query"]), 100, 5000, 3, 2.0)
        details.append({"query_id": record["query_id"], **score_ranking(rows, record)})
    report = {
        "mode": args.mode,
        "queries": len(records),
        "metrics": aggregate(details),
        "seconds": round(time.perf_counter() - started, 3),
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
