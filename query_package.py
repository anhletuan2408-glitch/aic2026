from __future__ import annotations

import csv
import io
import json
import re
import threading
from pathlib import Path
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

from submission import MAX_ANSWERS, validate_video_id
from trake_search import split_events


QUERY_TEXT = re.compile(r"^(query-.+-(kis|qa|trake))\.txt$")


class QueryPackage:
    def __init__(self, workspace: Path | None = None) -> None:
        self.lock = threading.Lock()
        self.workspace = workspace
        self.queries: dict[str, dict[str, object]] = {}
        self.outputs: dict[str, bytes] = {}
        self._load()

    def import_zip(self, payload: bytes) -> list[dict[str, object]]:
        if not payload or len(payload) > 5 * 1024 * 1024:
            raise ValueError("BTC package must be a non-empty ZIP no larger than 5 MB")
        found: dict[str, dict[str, object]] = {}
        try:
            with ZipFile(io.BytesIO(payload)) as archive:
                files = [entry for entry in archive.infolist() if not entry.is_dir()]
                if not files or len(files) > 100:
                    raise ValueError("BTC package must contain 1-100 query files")
                for entry in files:
                    if "/" in entry.filename or "\\" in entry.filename:
                        raise ValueError("Query files must be at the ZIP root")
                    match = QUERY_TEXT.fullmatch(entry.filename)
                    if not match:
                        raise ValueError(f"Invalid BTC query filename: {entry.filename}")
                    if entry.file_size > 64 * 1024:
                        raise ValueError(f"Query is too large: {entry.filename}")
                    text = archive.read(entry).decode("utf-8-sig").strip()
                    if not text:
                        raise ValueError(f"Empty query: {entry.filename}")
                    found[entry.filename] = {
                        "name": entry.filename,
                        "output_name": f"{match.group(1)}.csv",
                        "task": match.group(2),
                        "text": text,
                        "completed": False,
                        "rows": 0,
                    }
        except (BadZipFile, UnicodeDecodeError) as error:
            raise ValueError("Invalid ZIP or non-UTF-8 query file") from error
        with self.lock:
            self.queries = dict(sorted(found.items()))
            self.outputs = {}
            self._reset_workspace()
            self._persist()
            return self.status()

    def status(self) -> list[dict[str, object]]:
        return [dict(query) for query in self.queries.values()]

    def save(self, name: str, rows: list[dict[str, object]]) -> dict[str, object]:
        with self.lock:
            if name not in self.queries:
                raise ValueError(f"Unknown imported query: {name}")
            query = self.queries[name]
            content = self._csv_bytes(str(query["task"]), str(query["text"]), rows)
            self.outputs[name] = content
            query["completed"] = True
            query["rows"] = len(rows)
            if self.workspace is not None:
                (self.workspace / str(query["output_name"])).write_bytes(content)
            self._persist()
            return dict(query)

    def export_zip(self) -> bytes:
        with self.lock:
            if not self.queries:
                raise ValueError("Import a BTC query package first")
            missing = [name for name in self.queries if name not in self.outputs]
            if missing:
                raise ValueError("Incomplete queries: " + ", ".join(missing))
            stream = io.BytesIO()
            with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
                for name, query in self.queries.items():
                    archive.writestr(
                        f"submission/{query['output_name']}", self.outputs[name]
                    )
            return stream.getvalue()

    @staticmethod
    def _csv_bytes(task: str, query_text: str,
                   rows: list[dict[str, object]]) -> bytes:
        if not rows or len(rows) > MAX_ANSWERS:
            raise ValueError(f"A query result must contain 1-{MAX_ANSWERS} rows")
        expected_events = len(split_events(query_text)) if task == "trake" else None
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\r\n")
        for row in rows:
            video_id = str(row.get("video_id", ""))
            validate_video_id(video_id)
            if task == "trake":
                frames = row.get("frame_ids")
                if not isinstance(frames, list) or len(frames) != expected_events:
                    raise ValueError(f"TRAKE needs exactly {expected_events} frame IDs")
                frame_ids = [int(frame) for frame in frames]
                if any(frame < 0 for frame in frame_ids) or any(
                    left >= right for left, right in zip(frame_ids, frame_ids[1:])
                ):
                    raise ValueError("TRAKE frames must be non-negative and increasing")
                writer.writerow([video_id, *frame_ids])
            else:
                frame_idx = int(row.get("frame_idx", -1))
                if frame_idx < 0:
                    raise ValueError("frame_idx must be non-negative")
                if task == "qa":
                    answer = str(row.get("answer", ""))
                    if not answer or len(answer) > 100:
                        raise ValueError("Q&A answer must contain 1-100 characters")
                    writer.writerow([video_id, frame_idx, answer])
                else:
                    writer.writerow([video_id, frame_idx])
        return output.getvalue().encode("utf-8")
    def _reset_workspace(self) -> None:
        if self.workspace is None:
            return
        self.workspace.mkdir(parents=True, exist_ok=True)
        manifest = self.workspace / "package.json"
        if manifest.exists():
            manifest.unlink()
        for path in self.workspace.glob("query-*.csv"):
            if path.is_file():
                path.unlink()

    def _persist(self) -> None:
        if self.workspace is None:
            return
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / "package.json").write_text(
            json.dumps(self.status(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        if self.workspace is None:
            return
        manifest = self.workspace / "package.json"
        if not manifest.exists():
            return
        try:
            values = json.loads(manifest.read_text(encoding="utf-8"))
            self.queries = {str(item["name"]): item for item in values}
            for name, query in self.queries.items():
                path = self.workspace / str(query["output_name"])
                if bool(query.get("completed")) and path.is_file():
                    self.outputs[name] = path.read_bytes()
                elif bool(query.get("completed")):
                    query["completed"], query["rows"] = False, 0
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            self.queries, self.outputs = {}, {}
