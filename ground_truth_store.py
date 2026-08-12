from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from submission import validate_video_id

_ID = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


class GroundTruthStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()]

    @staticmethod
    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        query_id = str(payload.get("query_id", "")).strip()
        task = str(payload.get("task", "")).strip().casefold()
        query = str(payload.get("query", "")).strip()
        video_id = str(payload.get("video_id", "")).strip()
        if not _ID.fullmatch(query_id):
            raise ValueError("query_id must use 1-80 letters, numbers, dot, dash, or underscore")
        if task not in {"kis", "qa", "trake"}:
            raise ValueError("task must be kis, qa, or trake")
        if not query:
            raise ValueError("Ground-truth query must not be empty")
        validate_video_id(video_id)
        record: dict[str, Any] = {
            "query_id": query_id, "task": task, "query": query, "video_id": video_id,
        }
        if task in {"kis", "qa"}:
            start, end = int(payload.get("start", -1)), int(payload.get("end", -1))
            if start < 0 or end < start:
                raise ValueError("Ground-truth frame range is invalid")
            record.update(start=start, end=end)
        if task == "qa":
            answers = list(dict.fromkeys(
                str(value).strip() for value in payload.get("answers", []) if str(value).strip()
            ))
            if not answers or any(len(value) > 100 for value in answers):
                raise ValueError("QA ground truth needs 1-100 character accepted answers")
            record["answers"] = answers
        if task == "trake":
            moments = [[int(value[0]), int(value[1])] for value in payload.get("moments", [])]
            if len(moments) < 2 or any(start < 0 or end < start for start, end in moments):
                raise ValueError("TRAKE needs at least two valid [start,end] moments")
            record["moments"] = moments
        return record

    def upsert(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = self.validate(payload)
        with self.lock:
            records = self.records()
            index = next((i for i, item in enumerate(records)
                          if item["query_id"] == record["query_id"]), None)
            if index is None:
                records.append(record)
            else:
                records[index] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)
        return record