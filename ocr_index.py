from __future__ import annotations

import argparse
from contextlib import closing
import io
import os
import re
import sqlite3
import threading
import unicodedata
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Iterable

from PIL import Image

from web_app import KeyframeStore

_WORD = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
_LOCAL = threading.local()
_STOPWORDS = {
    "một", "những", "các", "đang", "được", "bị", "ở", "trong", "trên",
    "vào", "của", "có", "và", "với", "cho", "thì", "là", "đó", "này",
    "hình", "ảnh", "video", "khung", "frame", "hiển", "thị", "dòng", "chữ",
    "the", "a", "an", "in", "on", "of", "is", "are", "text", "screen",
}


def normalized_terms(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return [token for token in _WORD.findall(folded)
            if len(token) > 1 and token not in _STOPWORDS]


def longest_common_token_run(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    longest = 0
    for left_token in left:
        current = [0] * (len(right) + 1)
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current[index] = previous[index - 1] + 1
                longest = max(longest, current[index])
        previous = current
    return longest


def connect_ocr(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("""CREATE TABLE IF NOT EXISTS ocr_frames(
        global_id INTEGER PRIMARY KEY,
        video_id TEXT NOT NULL,
        keyframe_no INTEGER NOT NULL,
        frame_idx INTEGER NOT NULL,
        text TEXT NOT NULL,
        confidence REAL NOT NULL
    )""")
    connection.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts USING fts5(
        text, global_id UNINDEXED, tokenize='unicode61 remove_diacritics 2'
    )""")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_ocr_video ON ocr_frames(video_id)")
    return connection


def fts_query(text: str) -> str:
    tokens = normalized_terms(text)
    return " OR ".join(f'"{token.replace(chr(34), chr(34)*2)}"' for token in tokens[:24])


class OCRSignals:
    def __init__(self, path: Path) -> None:
        self.path = path

    def ranking(self, query: str, limit: int = 2500) -> list[int]:
        expression = fts_query(query)
        if not expression or not self.path.exists():
            return []
        ordered_query_terms = normalized_terms(query)
        query_terms = set(ordered_query_terms)
        with closing(sqlite3.connect(self.path, timeout=10.0)) as connection:
            rows = connection.execute(
                "SELECT CAST(global_id AS INTEGER),text,bm25(ocr_fts) FROM ocr_fts "
                "WHERE ocr_fts MATCH ? ORDER BY bm25(ocr_fts) LIMIT ?",
                (expression, max(limit * 2, limit)),
            ).fetchall()
        ranked = []
        for global_id, text, bm25_score in rows:
            text_terms = normalized_terms(str(text))
            overlap = len(query_terms.intersection(text_terms))
            coverage = overlap / max(len(query_terms), 1)
            phrase_run = longest_common_token_run(ordered_query_terms, text_terms)
            required_overlap = len(query_terms) if len(query_terms) <= 2 else 2
            if overlap >= required_overlap:
                ranked.append((int(global_id), phrase_run, overlap, coverage, float(bm25_score)))
        ranked.sort(key=lambda row: (-row[1], row[4], -row[2], -row[3]))
        return [global_id for global_id, _, _, _, _ in ranked[:limit]]

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with closing(sqlite3.connect(self.path, timeout=10.0)) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM ocr_frames").fetchone()[0])


def _engine():
    engine = getattr(_LOCAL, "engine", None)
    if engine is None:
        from rapidocr import RapidOCR
        engine = RapidOCR()
        _LOCAL.engine = engine
    return engine


def recognize(payload: bytes, min_confidence: float) -> tuple[str, float]:
    with Image.open(io.BytesIO(payload)) as source:
        image = source.convert("RGB")
        result = _engine()(image)
    texts = list(getattr(result, "txts", None) or [])
    scores = [float(value) for value in (getattr(result, "scores", None) or [])]
    kept = [(text.strip(), score) for text, score in zip(texts, scores)
            if text.strip() and score >= min_confidence]
    if not kept:
        return "", 0.0
    return " | ".join(text for text, _ in kept), sum(score for _, score in kept) / len(kept)


def metadata_rows(metadata: Path, completed: set[int], priority_stride: int,
                  limit: int | None) -> list[tuple[int, str, int, int]]:
    with closing(sqlite3.connect(metadata)) as connection:
        rows = connection.execute(
            "SELECT global_id,video_id,keyframe_no,frame_idx FROM keyframes "
            "ORDER BY CASE WHEN global_id % ? = 0 THEN 0 ELSE 1 END, global_id",
            (priority_stride,),
        ).fetchall()
    output = [tuple(row) for row in rows if int(row[0]) not in completed]
    return output[:limit] if limit is not None else output


def insert_batch(connection: sqlite3.Connection,
                 rows: list[tuple[int, str, int, int, str, float]]) -> None:
    with connection:
        connection.executemany(
            "INSERT OR REPLACE INTO ocr_frames VALUES(?,?,?,?,?,?)", rows
        )
        connection.executemany(
            "INSERT INTO ocr_fts(text,global_id) VALUES(?,?)",
            [(text, global_id) for global_id, _, _, _, text, _ in rows if text],
        )


def build(args: argparse.Namespace) -> None:
    output = connect_ocr(args.output)
    completed = {int(row[0]) for row in output.execute("SELECT global_id FROM ocr_frames")}
    rows = metadata_rows(args.metadata, completed, args.priority_stride, args.limit)
    store = KeyframeStore(args.zip_dir)
    started, processed, pending = time.perf_counter(), 0, {}
    batch: list[tuple[int, str, int, int, str, float]] = []
    max_pending = max(args.workers * 2, 2)

    def submit(pool: ThreadPoolExecutor, row: tuple[int, str, int, int]) -> None:
        global_id, video_id, keyframe_no, frame_idx = row
        payload = store.get_bytes(video_id, keyframe_no)
        pending[pool.submit(recognize, payload, args.min_confidence)] = row

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            iterator = iter(rows)
            exhausted = False
            while pending or not exhausted:
                while len(pending) < max_pending and not exhausted:
                    try:
                        submit(pool, next(iterator))
                    except StopIteration:
                        exhausted = True
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row = pending.pop(future)
                    text, confidence = future.result()
                    batch.append((*row, text, confidence))
                    processed += 1
                if len(batch) >= args.commit_every or (exhausted and not pending):
                    insert_batch(output, batch)
                    batch.clear()
                    elapsed = max(time.perf_counter() - started, 1e-6)
                    print(f"OCR {len(completed)+processed}/{len(completed)+len(rows)} "
                          f"({processed/elapsed:.2f} frame/s)", flush=True)
    finally:
        if batch:
            insert_batch(output, batch)
        store.close()
        output.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a resumable RapidOCR FTS5 index")
    parser.add_argument("--metadata", type=Path, default=Path("index/metadata.sqlite3"))
    parser.add_argument("--zip-dir", type=Path, default=Path("E:/"))
    parser.add_argument("--output", type=Path, default=Path("index/ocr.sqlite3"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--priority-stride", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--commit-every", type=int, default=25)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be in [1, 8]")
    if args.priority_stride < 1:
        parser.error("--priority-stride must be positive")
    return args


if __name__ == "__main__":
    build(parse_args())