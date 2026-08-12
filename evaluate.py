from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from submission import (
    FrameRange,
    KISAnswer,
    QAAnswer,
    TRAKEAnswer,
    final_score,
    kis_r_score,
    qa_exact_r_score,
    trake_r_score,
    validate_answers,
    validate_video_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one ranked AIC submission against local ground truth."
    )
    parser.add_argument("query_type", choices=["kis", "qna", "trake"])
    parser.add_argument("submission", type=Path)
    parser.add_argument("ground_truth", type=Path)
    return parser.parse_args()


def load_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.reader(stream) if row]
    validate_answers(rows)
    return rows


def parse_range(payload: dict[str, object]) -> FrameRange:
    return FrameRange(int(payload["start"]), int(payload["end"]))


def evaluate_kis(rows: list[list[str]], truth: dict[str, object]) -> list[float]:
    frame_range = parse_range(truth)
    video_id = str(truth["video_id"])
    validate_video_id(video_id)
    scores: list[float] = []
    for row in rows:
        if len(row) != 2:
            raise ValueError("KIS rows must be: <video_id>,<frame_id>")
        answer = KISAnswer(row[0], int(row[1]))
        validate_video_id(answer.video_id)
        scores.append(kis_r_score(answer, video_id, frame_range))
    return scores


def evaluate_qna(rows: list[list[str]], truth: dict[str, object]) -> list[float]:
    frame_range = parse_range(truth)
    video_id = str(truth["video_id"])
    accepted_answers = [str(value) for value in truth["answers"]]
    validate_video_id(video_id)
    scores: list[float] = []
    for row in rows:
        if len(row) != 3:
            raise ValueError("Q&A rows must be: <video_id>,<frame_id>,<answer>")
        answer = QAAnswer(row[0], int(row[1]), row[2])
        validate_video_id(answer.video_id)
        scores.append(max(
            qa_exact_r_score(answer, video_id, frame_range, accepted)
            for accepted in accepted_answers
        ))
    return scores


def evaluate_trake(rows: list[list[str]], truth: dict[str, object]) -> list[float]:
    video_id = str(truth["video_id"])
    ranges = [FrameRange(int(item[0]), int(item[1])) for item in truth["moments"]]
    validate_video_id(video_id)
    scores: list[float] = []
    for row in rows:
        if len(row) != len(ranges) + 1:
            raise ValueError(
                f"TRAKE rows must contain video_id plus {len(ranges)} frame IDs"
            )
        answer = TRAKEAnswer(row[0], tuple(int(value) for value in row[1:]))
        validate_video_id(answer.video_id)
        scores.append(trake_r_score(answer, video_id, ranges))
    return scores


def main() -> None:
    args = parse_args()
    rows = load_rows(args.submission)
    truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    evaluators = {
        "kis": evaluate_kis,
        "qna": evaluate_qna,
        "trake": evaluate_trake,
    }
    scores = evaluators[args.query_type](rows, truth)
    top_scores, score = final_score(scores)
    for k, value in top_scores.items():
        print(f"R@{k}: {value:.6f}")
    print(f"Final Score: {score:.6f}")


if __name__ == "__main__":
    main()
