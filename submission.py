from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


MAX_ANSWERS = 100
TOP_THRESHOLDS = (1, 5, 20, 50, 100)
VIDEO_ID_PATTERN = re.compile(r"^L\d{2}_V\d{3}$")


@dataclass(frozen=True)
class FrameRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid frame range: [{self.start}, {self.end}]")

    def contains(self, frame_id: int) -> bool:
        return self.start <= frame_id <= self.end


@dataclass(frozen=True)
class KISAnswer:
    video_id: str
    frame_id: int


@dataclass(frozen=True)
class QAAnswer:
    video_id: str
    frame_id: int
    answer: str


@dataclass(frozen=True)
class TRAKEAnswer:
    video_id: str
    frame_ids: tuple[int, ...]


def validate_video_id(video_id: str) -> None:
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError(f"Invalid video_id: {video_id!r}")


def validate_answers(answers: Sequence[object]) -> None:
    if not answers:
        raise ValueError("A submission must contain at least one answer")
    if len(answers) > MAX_ANSWERS:
        raise ValueError(f"A submission may contain at most {MAX_ANSWERS} answers")


def normalize_semantic_answer(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return " ".join(normalized.split())


def kis_r_score(
    answer: KISAnswer, ground_truth_video: str, ground_truth_range: FrameRange
) -> float:
    return float(
        answer.video_id == ground_truth_video
        and ground_truth_range.contains(answer.frame_id)
    )


def qa_r_score(
    answer: QAAnswer,
    ground_truth_video: str,
    ground_truth_range: FrameRange,
    accepted_answers: Sequence[str],
) -> float:
    accepted = {normalize_semantic_answer(value) for value in accepted_answers}
    return float(
        answer.video_id == ground_truth_video
        and ground_truth_range.contains(answer.frame_id)
        and normalize_semantic_answer(answer.answer) in accepted
    )


def trake_r_score(
    answer: TRAKEAnswer,
    ground_truth_video: str,
    ground_truth_ranges: Sequence[FrameRange],
) -> float:
    if answer.video_id != ground_truth_video:
        return 0.0
    if not ground_truth_ranges:
        raise ValueError("TRAKE ground truth must contain at least one moment")
    if len(answer.frame_ids) != len(ground_truth_ranges):
        raise ValueError(
            "TRAKE answer and ground truth must contain the same number of moments"
        )
    matched = sum(
        frame_range.contains(frame_id)
        for frame_id, frame_range in zip(answer.frame_ids, ground_truth_ranges)
    )
    return matched / len(ground_truth_ranges)


def r_at_k(scores: Sequence[float], k: int) -> float:
    if not scores:
        return 0.0
    return max(scores[: min(k, len(scores))])


def final_score(scores: Sequence[float]) -> tuple[dict[int, float], float]:
    validate_answers(scores)
    if any(score < 0.0 or score > 1.0 for score in scores):
        raise ValueError("Every R-Score must be in [0, 1]")
    values = {k: r_at_k(scores, k) for k in TOP_THRESHOLDS}
    return values, sum(values.values()) / len(TOP_THRESHOLDS)


def write_kis_submission(path: Path, answers: Sequence[KISAnswer]) -> None:
    validate_answers(answers)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for answer in answers:
            validate_video_id(answer.video_id)
            if answer.frame_id < 0:
                raise ValueError("frame_id must be non-negative")
            writer.writerow([answer.video_id, answer.frame_id])
