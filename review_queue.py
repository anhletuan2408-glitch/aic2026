from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
import re
import unicodedata


@dataclass(frozen=True)
class ReviewDecision:
    confidence: float
    required: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def _answer_key(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()
    number_aliases = {
        "khong nguoi": "0", "zero people": "0",
        "mot nguoi": "1", "one person": "1",
        "hai nguoi": "2", "two people": "2",
        "ba nguoi": "3", "three people": "3",
        "bon nguoi": "4", "four people": "4",
        "nam nguoi": "5", "five people": "5",
        "sau nguoi": "6", "six people": "6",
        "bay nguoi": "7", "seven people": "7",
        "tam nguoi": "8", "eight people": "8",
        "chin nguoi": "9", "nine people": "9",
        "muoi nguoi": "10", "ten people": "10",
    }
    folded = "".join(
        char for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    return number_aliases.get(folded, text)


def assess_results(task: str, rows: list[dict[str, object]],
                   threshold: float = 0.74) -> ReviewDecision:
    """Estimate which auto-produced query results deserve scarce human time."""
    task = task.casefold().strip()
    if not rows:
        return ReviewDecision(0.0, True, "No automatic result")

    if task == "kis":
        scores = []
        for row in rows[:5]:
            try:
                scores.append(float(row["score"]))
            except (KeyError, TypeError, ValueError):
                break
        if len(scores) >= 2 and abs(scores[0]) > 1e-9:
            margin = max(0.0, (scores[0] - scores[1]) / abs(scores[0]))
            top_video = str(rows[0].get("video_id", ""))
            support = sum(
                str(row.get("video_id", "")) == top_video for row in rows[:5]
            ) / min(5, len(rows))
            confidence = _clamp(0.42 + 0.38 * min(1.0, margin * 5) + 0.20 * support)
            reason = "Weak Top-1 margin or video consensus"
        else:
            confidence = 0.45
            reason = "No calibrated retrieval score"

    elif task == "qa":
        first_by_frame: list[str] = []
        seen_frames: set[tuple[str, int]] = set()
        for row in rows[:30]:
            key = (str(row.get("video_id", "")), int(row.get("frame_idx", -1)))
            if key in seen_frames:
                continue
            seen_frames.add(key)
            answer = _answer_key(row.get("answer", ""))
            if answer:
                first_by_frame.append(answer)
            if len(first_by_frame) == 8:
                break
        if not first_by_frame:
            confidence = 0.0
        else:
            consensus = Counter(first_by_frame).most_common(1)[0][1] / len(first_by_frame)
            coverage = min(1.0, len(first_by_frame) / 5)
            confidence = _clamp(0.28 + 0.58 * consensus + 0.14 * coverage)
        reason = "Qwen answers disagree across candidate frames"

    elif task == "trake":
        top_video = str(rows[0].get("video_id", ""))
        sample = rows[:5]
        support = sum(str(row.get("video_id", "")) == top_video for row in sample) / len(sample)
        lengths = [len(row.get("frame_ids", [])) for row in sample
                   if isinstance(row.get("frame_ids"), list)]
        valid = bool(lengths) and len(set(lengths)) == 1 and lengths[0] > 1
        confidence = _clamp(0.38 + 0.42 * support + (0.12 if valid else 0.0))
        reason = "Temporal paths disagree on the target video"
    else:
        raise ValueError("task must be kis, qa, or trake")

    return ReviewDecision(confidence, confidence < threshold, reason)
