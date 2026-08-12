from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from submission import MAX_ANSWERS, validate_video_id


QUERY_FILE = re.compile(r"^query-.+-(kis|qa|trake)\.csv$")


def parse_frame(value: str, location: str) -> int:
    if value != value.strip() or not value.isdigit():
        raise ValueError(f"Invalid integer frame at {location}: {value!r}")
    return int(value)


def validate_query_csv(path: Path) -> tuple[str, int]:
    match = QUERY_FILE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Invalid query filename: {path.name}")
    query_type = match.group(1)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows or len(rows) > MAX_ANSWERS:
        raise ValueError(f"{path.name} must contain 1-{MAX_ANSWERS} rows")
    trake_width: int | None = None
    for row_number, row in enumerate(rows, start=1):
        location = f"{path.name}:{row_number}"
        expected = {"kis": 2, "qa": 3}.get(query_type)
        if expected is not None and len(row) != expected:
            raise ValueError(f"{location} must contain {expected} columns")
        if query_type == "trake" and len(row) < 3:
            raise ValueError(f"{location} must contain a video and at least 2 events")
        if query_type == "trake":
            trake_width = trake_width or len(row)
            if len(row) != trake_width:
                raise ValueError(f"{path.name} TRAKE rows must have equal event counts")
        validate_video_id(row[0])
        frames = [
            parse_frame(value, location)
            for value in (row[1:] if query_type == "trake" else row[1:2])
        ]
        if query_type == "trake" and any(a >= b for a, b in zip(frames, frames[1:])):
            raise ValueError(f"{location} TRAKE frames must be strictly increasing")
        if query_type == "qa" and (not row[2] or len(row[2]) > 100):
            raise ValueError(f"{location} answer must contain 1-100 characters")
    return query_type, len(rows)


def build_submission_zip(source: Path, output: Path) -> list[tuple[str, str, int]]:
    files = sorted(source.glob("query-*.csv"))
    if not files:
        raise ValueError(f"No query CSV files found in {source}")
    report = [(path.name, *validate_query_csv(path)) for path in files]
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, f"submission/{path.name}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate AIC query CSV files and create submission/*.csv ZIP"
    )
    parser.add_argument("source", type=Path, help="Directory containing query-*.csv")
    parser.add_argument("output", type=Path, help="Output .zip path")
    args = parser.parse_args()
    if args.output.suffix.casefold() != ".zip":
        raise ValueError("Output must have a .zip extension")
    report = build_submission_zip(args.source, args.output)
    for name, query_type, rows in report:
        print(f"OK {name}: {query_type}, {rows} rows")
    print(f"Wrote {args.output} with submission/ as the archive root")


if __name__ == "__main__":
    main()
