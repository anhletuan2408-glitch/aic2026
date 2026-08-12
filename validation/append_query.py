from __future__ import annotations

import argparse
import json
from pathlib import Path

from submission import validate_video_id


def make_record(
    query_id: str, query: str, video_id: str, start: int, end: int, round_name: str
) -> dict[str, object]:
    query_id = query_id.strip()
    query = " ".join(query.split())
    round_name = round_name.strip()
    if not query_id or not query or not round_name:
        raise ValueError("query_id, query, and round must not be blank")
    validate_video_id(video_id)
    if start < 0 or end < start:
        raise ValueError("frame range must satisfy 0 <= start <= end")
    return {
        "query_id": query_id,
        "query": query,
        "video_id": video_id,
        "start": start,
        "end": end,
        "round": round_name,
    }


def append_record(path: Path, record: dict[str, object]) -> None:
    existing_ids: set[str] = set()
    if path.exists():
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    existing_ids.add(str(json.loads(line)["query_id"]))
    if str(record["query_id"]) in existing_ids:
        raise ValueError(f"Duplicate query_id: {record['query_id']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append one manually verified KIS query")
    parser.add_argument("path", type=Path)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--round", dest="round_name", required=True)
    args = parser.parse_args()
    record = make_record(
        args.query_id, args.query, args.video_id,
        args.start, args.end, args.round_name,
    )
    append_record(args.path, record)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
