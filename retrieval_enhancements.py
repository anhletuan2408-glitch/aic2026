from __future__ import annotations

import re

import numpy as np


VIETNAMESE_FILLERS = {
    "một", "những", "các", "đang", "được", "bị", "ở", "trong", "trên",
    "vào", "của", "có", "và", "với", "cho", "thì", "là", "đó", "này",
}


_QA_QUESTION_PATTERNS = (
    r"\b(?:c\u00f3\s+)?bao\s+nhi\u00eau\b",
    r"\b(?:c\u00f3\s+)?m\u00e0u(?:\s+s\u1eafc)?\s+(?:g\u00ec|n\u00e0o)\b",
    r"\bhow\s+many\b",
    r"\b(?:g\u00ec|n\u00e0o|ai)\b",
    r"\b(?:what|which|who)\b",
)


def qa_retrieval_query(question: str) -> str:
    original = " ".join(question.split())
    if not original:
        raise ValueError("Question must not be empty")
    cleaned = original
    for pattern in _QA_QUESTION_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.strip(" ,.;:!?").split())
    return cleaned if len(cleaned.split()) >= 3 else original


def expand_query(query: str, max_variants: int = 3) -> list[str]:
    original = " ".join(query.split())
    if not original:
        raise ValueError("Query must not be empty")
    variants = [original]
    words = original.split()
    if len(words) < 7:
        return variants
    content = " ".join(
        word for word in words
        if word.casefold().strip(",.;:!?") not in VIETNAMESE_FILLERS
    )
    if len(content.split()) >= 3 and content.casefold() != original.casefold():
        variants.append(content)
    known = {item.casefold() for item in variants}
    for clause in re.split(r"[,;.!?]+|\s+và\s+", original, flags=re.IGNORECASE):
        clause = clause.strip()
        if len(clause.split()) >= 3 and clause.casefold() not in known:
            variants.append(clause)
            known.add(clause.casefold())
        if len(variants) >= max_variants:
            break
    return variants[:max_variants]


def fuse_query_rankings(
    rankings: list[list[int]], weights: list[float] | None = None,
    rrf_k: int = 60, limit: int | None = None,
) -> tuple[list[int], np.ndarray]:
    if not rankings:
        return [], np.empty(0, dtype=np.float32)
    weights = weights or [1.0, *([0.10] * (len(rankings) - 1))]
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must have equal length")
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, global_id in enumerate(ranking, start=1):
            scores[global_id] = scores.get(global_id, 0.0) + weight / (rrf_k + rank)
    ordered = sorted(scores, key=scores.get, reverse=True)
    if limit is not None:
        ordered = ordered[:limit]
    return ordered, np.asarray([scores[value] for value in ordered], dtype=np.float32)


def temporal_neighbor_ranking(
    seed_ids: list[int], video_by_id: dict[int, str], frame_count: int,
    seed_count: int = 500, radius: int = 1,
) -> list[int]:
    neighbors: list[int] = []
    seen: set[int] = set()
    for seed in seed_ids[:seed_count]:
        video_id = video_by_id.get(seed)
        for distance in range(1, radius + 1):
            for candidate in (seed - distance, seed + distance):
                if not 0 <= candidate < frame_count or candidate in seen:
                    continue
                if video_by_id.get(candidate) != video_id:
                    continue
                seen.add(candidate)
                neighbors.append(candidate)
    return neighbors


def raw_temporal_neighbor_ids(
    seed_ids: list[int], frame_count: int, seed_count: int = 500, radius: int = 1
) -> list[int]:
    return list(dict.fromkeys(
        candidate
        for seed in seed_ids[:seed_count]
        for distance in range(1, radius + 1)
        for candidate in (seed - distance, seed + distance)
        if 0 <= candidate < frame_count
    ))
