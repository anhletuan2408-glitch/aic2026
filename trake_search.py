from __future__ import annotations

import argparse
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from retrieval_enhancements import fuse_query_rankings, temporal_event_variants
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
    beam_width: int = 64,
    max_answers: int = 100,
    gap_profiles: tuple[float, ...] = (0.0, 15.0, 45.0, 120.0, 240.0),
    priority_videos: int = 5,
) -> list[tuple[float, TRAKEAnswer]]:
    """Return diverse k-best ordered paths using dynamic programming per video."""
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
    def completed_paths(
        video_id: str, minimum_gap: float,
    ) -> list[tuple[float, TRAKEAnswer]]:
        states: list[tuple[EventCandidate, float, tuple[int, ...]]] = [
            (candidate, candidate.score, (candidate.frame_idx,))
            for candidate in by_event[0][video_id]
        ]
        states.sort(key=lambda item: item[1], reverse=True)
        states = states[:beam_width]
        for grouped in by_event[1:]:
            next_states: list[tuple[EventCandidate, float, tuple[int, ...]]] = []
            for candidate in grouped[video_id]:
                extensions = []
                for previous, score, frames in states:
                    if candidate.frame_idx <= previous.frame_idx:
                        continue
                    gap = max(0.0, candidate.pts_time - previous.pts_time)
                    if gap < minimum_gap:
                        continue
                    distinct_bonus = min(gap, 10.0) * 0.0005
                    continuity_penalty = max(0.0, gap - 15.0) * 0.00025
                    extensions.append((
                        candidate,
                        score + candidate.score + distinct_bonus - continuity_penalty,
                        (*frames, candidate.frame_idx),
                    ))
                extensions.sort(key=lambda item: item[1], reverse=True)
                next_states.extend(extensions[:beam_width])
            # Keep one global beam, rather than one beam per candidate.  The old
            # form could grow to per_video * beam_width states at every event
            # and exhaust RAM for ordinary three-event queries.
            next_states.sort(key=lambda item: item[1], reverse=True)
            states = next_states[:beam_width]
            if not states:
                break
        completed = [
            (score, TRAKEAnswer(video_id, frames))
            for _, score, frames in states if len(frames) == len(event_rows)
        ]
        completed.sort(key=lambda item: item[0], reverse=True)
        return completed

    aligned: list[tuple[float, TRAKEAnswer]] = []
    profile_paths: dict[str, list[tuple[float, TRAKEAnswer]]] = {}
    best_by_video: dict[str, float] = {}
    for video_id in common_videos:
        completed = completed_paths(video_id, gap_profiles[0])
        if not completed:
            continue
        aligned.extend(completed)
        profile_paths[video_id] = [completed[0]]
        best_by_video[video_id] = completed[0][0]

    priority = sorted(
        best_by_video, key=best_by_video.__getitem__, reverse=True,
    )[:priority_videos]
    for video_id in priority:
        for minimum_gap in gap_profiles[1:]:
            completed = completed_paths(video_id, minimum_gap)
            if completed:
                profile_paths[video_id].append(completed[0])
    aligned.sort(key=lambda item: item[0], reverse=True)
    output: list[tuple[float, TRAKEAnswer]] = []
    seen_paths: set[tuple[str, tuple[int, ...]]] = set()

    def append_unique(item: tuple[float, TRAKEAnswer]) -> None:
        key = (item[1].video_id, item[1].frame_ids)
        if key not in seen_paths and len(output) < max_answers:
            seen_paths.add(key)
            output.append(item)

    for video_id in priority:
        variants = profile_paths[video_id]
        append_unique(variants[0])
        if len(variants) > 1:
            append_unique(variants[-1])
    for profile_index in range(1, max(1, len(gap_profiles) - 1)):
        for video_id in priority:
            variants = profile_paths[video_id]
            if profile_index < len(variants) - 1:
                append_unique(variants[profile_index])

    seen_videos = {item[1].video_id for item in output}
    for item in aligned:
        if item[1].video_id not in seen_videos:
            seen_videos.add(item[1].video_id)
            append_unique(item)
        if len(seen_videos) >= min(50, len(best_by_video)):
            break
    for item in aligned:
        append_unique(item)
    return output[:max_answers]


def joint_video_candidates(
    event_rows: list[list[dict[str, object]]], limit: int = 80,
) -> list[str]:
    """Rank videos supported by every event before the conditioned second pass."""
    if not event_rows or limit <= 0:
        return []
    evidence: list[dict[str, float]] = []
    for rows in event_rows:
        best: dict[str, float] = {}
        for rank, row in enumerate(rows, start=1):
            video_id = str(row["video_id"])
            relevance = float(row.get("score", -2.0))
            score = (relevance if relevance >= -1.0 else 0.0) + 1.0 / (60 + rank)
            best[video_id] = max(best.get(video_id, -float("inf")), score)
        evidence.append(best)
    common = set(evidence[0])
    for best in evidence[1:]:
        common.intersection_update(best)
    return sorted(
        common, key=lambda video_id: sum(best[video_id] for best in evidence),
        reverse=True,
    )[:limit]


def fuse_event_rankings(
    base_ids: np.ndarray,
    base_scores: np.ndarray,
    siglip_ids: np.ndarray | None = None,
    limit: int = 10000,
) -> list[tuple[int, float]]:
    """Fuse one event's dense rankings while preserving legacy scores if alone."""
    base = [
        (int(global_id), float(score))
        for global_id, score in zip(base_ids, base_scores)
        if global_id >= 0
    ]
    if siglip_ids is None:
        return base[:limit]
    rankings = [
        [global_id for global_id, _ in base],
        [int(global_id) for global_id in siglip_ids if global_id >= 0],
    ]
    fused_ids, _ = fuse_query_rankings(
        rankings, weights=[1.0, 1.0], limit=limit
    )
    # Cross-model similarities are not calibrated. The sentinel makes
    # downstream alignment use reciprocal rank instead of raw similarity.
    return [(global_id, -2.0) for global_id in fused_ids]


def add_video_conditioned_candidates(
    engine: MultilingualFaissEngine,
    event_vectors: np.ndarray,
    event_rows: list[list[dict[str, object]]],
    video_ids: list[str],
    per_video: int,
    min_time_gap: float = 20.0,
) -> None:
    """Search every frame in jointly supported videos, then merge per-event rows."""
    if not video_ids:
        return
    placeholders = ",".join("?" for _ in video_ids)
    with closing(sqlite3.connect(engine.metadata_path)) as connection:
        records = connection.execute(
            f"SELECT global_id,video_id,frame_idx,pts_time FROM keyframes "
            f"WHERE video_id IN ({placeholders}) ORDER BY global_id",
            video_ids,
        ).fetchall()
    by_video: dict[str, list[tuple[int, int, float]]] = {}
    for global_id, video_id, frame_idx, pts_time in records:
        by_video.setdefault(str(video_id), []).append(
            (int(global_id), int(frame_idx), float(pts_time))
        )
    additions: list[list[dict[str, object]]] = [[] for _ in event_rows]
    rank_only = [
        bool(rows) and all(float(row.get("score", -2.0)) < -1.0 for row in rows)
        for rows in event_rows
    ]
    for video_id in video_ids:
        video_records = by_video.get(video_id, [])
        if not video_records:
            continue
        global_ids = np.asarray([row[0] for row in video_records], dtype=np.int64)
        frame_vectors = engine.index.reconstruct_batch(global_ids)
        scores = np.asarray(event_vectors @ frame_vectors.T, dtype=np.float32)
        count = min(per_video, len(video_records))
        for event_index, event_scores in enumerate(scores):
            selected: list[int] = []
            for index in np.argsort(event_scores)[::-1]:
                pts_time = video_records[int(index)][2]
                if any(
                    abs(pts_time - video_records[other][2]) < min_time_gap
                    for other in selected
                ):
                    continue
                selected.append(int(index))
                if len(selected) >= count:
                    break
            for index in selected:
                _, frame_idx, pts_time = video_records[int(index)]
                additions[event_index].append({
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "pts_time": pts_time,
                    "score": (
                        -2.0 if rank_only[event_index]
                        else float(event_scores[int(index)])
                    ),
                })
    for rows, extra in zip(event_rows, additions):
        merged = {
            (str(row["video_id"]), int(row["frame_idx"])): row for row in rows
        }
        for row in extra:
            key = (str(row["video_id"]), int(row["frame_idx"]))
            if key not in merged or float(row["score"]) > float(merged[key]["score"]):
                merged[key] = row
        rows[:] = sorted(
            merged.values(), key=lambda row: float(row.get("score", -2.0)),
            reverse=True,
        )


def search_trake(
    engine: MultilingualFaissEngine,
    events: list[str],
    candidate_k: int = 4000,
    per_video: int = 32,
) -> list[TRAKEAnswer]:
    # Bound FAISS result buffers on the 4 GB target machine.  Recall is
    # recovered by object fusion and the video-conditioned second pass below.
    candidate_k = max(200, min(int(candidate_k), 4000))
    per_video = max(4, min(int(per_video), 32))
    variant_groups = [temporal_event_variants(event) for event in events]
    flattened = [variant for group in variant_groups for variant in group]
    flat_vectors = engine.encode_many(flattened)
    vectors = []
    offset = 0
    for group in variant_groups:
        combined = flat_vectors[offset:offset + len(group)].mean(axis=0)
        combined /= max(float(np.linalg.norm(combined)), 1e-8)
        vectors.append(combined)
        offset += len(group)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    similarities, ids = engine.index.search(
        vectors, min(candidate_k, engine.index.ntotal)
    )
    siglip_ids: np.ndarray | None = None
    global_siglip = getattr(engine, "siglip2_index", None)
    reranker = getattr(engine, "reranker", None)
    if global_siglip is not None and reranker is not None:
        flat_siglip = reranker.encode_text_many(flattened)
        siglip_vectors = []
        offset = 0
        for group in variant_groups:
            combined = flat_siglip[offset:offset + len(group)].mean(axis=0)
            combined /= max(float(np.linalg.norm(combined)), 1e-8)
            siglip_vectors.append(combined)
            offset += len(group)
        siglip_vectors = np.ascontiguousarray(siglip_vectors, dtype=np.float32)
        _, siglip_ids = global_siglip.search(
            siglip_vectors, min(candidate_k, global_siglip.ntotal)
        )
    rows: list[list[dict[str, object]]] = []
    for event_index, (event_ids, event_scores) in enumerate(zip(ids, similarities)):
        event_siglip_ids = None if siglip_ids is None else siglip_ids[event_index]
        ranked_pairs = fuse_event_rankings(
            event_ids, event_scores, event_siglip_ids, candidate_k
        )
        hybrid = getattr(engine, "hybrid", None)
        if hybrid is not None:
            object_ids, _ = hybrid.object_conjunction_ranking(
                events[event_index], engine.hybrid_config
            )
            if object_ids:
                fused_ids, _ = fuse_query_rankings(
                    [[global_id for global_id, _ in ranked_pairs], object_ids],
                    weights=[1.0, 0.80], limit=candidate_k,
                )
                ranked_pairs = [(global_id, -2.0) for global_id in fused_ids]
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
    joint_videos = joint_video_candidates(rows)
    add_video_conditioned_candidates(
        engine, vectors, rows, joint_videos, per_video
    )
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
