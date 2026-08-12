from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from submission import TRAKEAnswer, write_trake_submission
from search import load_metadata
if TYPE_CHECKING:
    from web_app_vi import MultilingualFaissEngine
else:
    MultilingualFaissEngine = Any


@dataclass(frozen=True)
class EventCandidate:
    video_id: str
    frame_idx: int
    rank: int
    relevance: float
    pts_time: float

    @property
    def score(self) -> float:
        return self.relevance if self.relevance >= 0.0 else 1.0 / (60 + self.rank)

def split_events(text: str) -> list[str]:
    text = text.strip()
    if not text:
        raise ValueError("TRAKE query must not be empty")
    lines = [
        re.sub(r"^\s*(?:event|sự kiện)?\s*\d+\s*[:.)-]\s*", "", line,
               flags=re.IGNORECASE).strip()
        for line in text.splitlines() if line.strip()
    ]
    if len(lines) > 1:
        events = lines
    else:
        events = [part.strip() for part in re.split(r"\s*(?:->|→|;)\s*", text)]
    events = [event for event in events if event]
    if len(events) < 2:
        raise ValueError(
            "TRAKE requires at least two events; separate them by new lines, ';', or '->'"
        )
    return events


def align_event_candidates(
    event_rows: list[list[dict[str, object]]],
    beam_width: int = 8,
    max_answers: int = 100,
) -> list[tuple[float, TRAKEAnswer]]:
    """Return k-best ordered event paths using dynamic programming per video."""
    by_event: list[dict[str, list[EventCandidate]]] = []
    for rows in event_rows:
        grouped: dict[str, list[EventCandidate]] = {}
        for rank, row in enumerate(rows, start=1):
            raw_score = float(row.get("score", -2.0))
            relevance = (raw_score + 1.0) / 2.0 if raw_score >= -1.0 else -1.0
            candidate = EventCandidate(
                str(row["video_id"]), int(row["frame_idx"]), rank,
                relevance, float(row.get("pts_time", row["frame_idx"])),
            )
            grouped.setdefault(candidate.video_id, []).append(candidate)
        for candidates in grouped.values():
            candidates.sort(key=lambda item: item.frame_idx)
        by_event.append(grouped)
    common_videos = set(by_event[0])
    for grouped in by_event[1:]:
        common_videos.intersection_update(grouped)
    aligned: list[tuple[float, TRAKEAnswer]] = []
    for video_id in common_videos:
        states: list[tuple[EventCandidate, float, tuple[int, ...]]] = [
            (candidate, candidate.score, (candidate.frame_idx,))
            for candidate in by_event[0][video_id]
        ]
        for grouped in by_event[1:]:
            next_states: list[tuple[EventCandidate, float, tuple[int, ...]]] = []
            for candidate in grouped[video_id]:
                extensions = []
                for previous, score, frames in states:
                    if candidate.frame_idx <= previous.frame_idx:
                        continue
                    gap = max(0.0, candidate.pts_time - previous.pts_time)
                    distinct_bonus = min(gap, 10.0) * 0.0005
                    extensions.append((candidate, score + candidate.score + distinct_bonus,
                                       (*frames, candidate.frame_idx)))
                extensions.sort(key=lambda item: item[1], reverse=True)
                next_states.extend(extensions[:beam_width])
            states = next_states
            if not states:
                break
        aligned.extend(
            (score, TRAKEAnswer(video_id, frames))
            for _, score, frames in states if len(frames) == len(event_rows)
        )
    aligned.sort(key=lambda item: item[0], reverse=True)
    return aligned[:max_answers]
def search_trake(
    engine: MultilingualFaissEngine,
    events: list[str],
    candidate_k: int = 10000,
    per_video: int = 25,
) -> list[TRAKEAnswer]:
    vectors = engine.encode_many(events)
    similarities, ids = engine.index.search(vectors, min(candidate_k, engine.index.ntotal))
    rows: list[list[dict[str, object]]] = []
    for event_ids, event_scores in zip(ids, similarities):
        ranked_pairs = [(int(value), float(score)) for value, score in zip(event_ids, event_scores) if value >= 0]
        ranked_ids = [value for value, _ in ranked_pairs]
        score_by_id = dict(ranked_pairs)
        metadata = load_metadata(engine.metadata_path, ranked_ids)
        counts: dict[str, int] = {}
        event_rows: list[dict[str, object]] = []
        for global_id in ranked_ids:
            row = metadata[global_id]
            video_id = str(row["video_id"])
            if counts.get(video_id, 0) >= per_video:
                continue
            counts[video_id] = counts.get(video_id, 0) + 1
            event_rows.append({
                "video_id": video_id,
                "frame_idx": int(row["frame_idx"]),
                "pts_time": float(row["pts_time"]),
                "score": score_by_id[global_id],
            })
        rows.append(event_rows)
    return [answer for _, answer in align_event_candidates(rows)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve and align ordered TRAKE events")
    parser.add_argument("query", help="Query text or path to a UTF-8 query file")
    parser.add_argument("--event", action="append", default=[])
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = Path(args.query)
    query_text = path.read_text(encoding="utf-8-sig") if path.is_file() else args.query
    events = args.event or split_events(query_text)
    from web_app_vi import MultilingualFaissEngine as Engine
    engine = Engine(args.index_dir, args.device)
    answers = search_trake(engine, events)
    if not answers:
        raise RuntimeError("No video contained an increasing candidate sequence for all events")
    write_trake_submission(args.output, answers)
    print(f"Events: {len(events)}")
    print(f"Wrote {len(answers)} TRAKE sequences to {args.output}")


if __name__ == "__main__":
    main()
