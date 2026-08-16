from __future__ import annotations

import argparse
import gc
import io
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from submission import QAAnswer, write_qa_submission
from multicrop import crop_image
from text_encoding import repair_utf8_mojibake
from web_app import KeyframeStore
if TYPE_CHECKING:
    from web_app_vi import MultilingualFaissEngine
else:
    MultilingualFaissEngine = Any


QWEN_VL_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

_COUNT_ALIASES = {
    "0": ("0", "không người", "zero people"), "1": ("1", "một người", "one person"),
    "2": ("2", "hai người", "two people"), "3": ("3", "ba người", "three people"),
    "4": ("4", "bốn người", "four people"), "5": ("5", "năm người", "five people"),
    "6": ("6", "sáu người", "six people"), "7": ("7", "bảy người", "seven people"),
    "8": ("8", "tám người", "eight people"), "9": ("9", "chín người", "nine people"),
    "10": ("10", "mười người", "ten people"),
}
_COLOR_ALIASES = {
    "đỏ": ("đỏ", "màu đỏ", "red"), "xanh dương": ("xanh dương", "màu xanh dương", "blue"),
    "xanh lá": ("xanh lá", "màu xanh lá", "green"), "vàng": ("vàng", "màu vàng", "yellow"),
    "đen": ("đen", "màu đen", "black"), "trắng": ("trắng", "màu trắng", "white"),
    "hồng": ("hồng", "màu hồng", "pink"), "cam": ("cam", "màu cam", "orange"),
    "tím": ("tím", "màu tím", "purple"),
}


def clean_answer(value: str) -> str:
    value = repair_utf8_mojibake(value).strip().splitlines()[0].strip()
    value = re.sub(r"^(?:answer|trả lời)\s*:\s*", "", value, flags=re.IGNORECASE)
    value = value.strip().strip('"').strip()
    if value.endswith((".", "!", "?")):
        value = value[:-1].rstrip()
    if not value:
        raise ValueError("VLM returned an empty answer")
    return value[:100]


def answer_aliases(question: str, value: str) -> list[str]:
    """Return conservative exact-match aliases without changing visual meaning."""
    answer = clean_answer(value)
    aliases = [answer]
    folded_question, folded_answer = question.casefold(), answer.casefold()
    if re.search(r"\btên\b|\bứng viên\b|\bwho\b|\bname\b", folded_question):
        shortened = re.sub(r"(?:\s*[-–|:]?\s*)\b(?:19|20)\d{2}\b.*$", "", answer).strip()
        if shortened and shortened.casefold() != folded_answer:
            aliases.insert(0, shortened)
    if re.search(r"\bbao nhiêu\b|\bhow many\b", folded_question):
        for digit, variants in _COUNT_ALIASES.items():
            if any(re.search(r"\b" + re.escape(item.casefold()) + r"\b", folded_answer) for item in variants):
                aliases.extend((*variants, digit))
                break
    if re.search(r"\bmàu\b|\bcolor\b", folded_question):
        for variants in _COLOR_ALIASES.values():
            if any(re.search(r"\b" + re.escape(item.casefold()) + r"\b", folded_answer) for item in variants):
                aliases.extend(variants)
                break
    output, seen = [], set()
    for alias in aliases:
        cleaned = clean_answer(alias)
        if cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            output.append(cleaned)
    return output


def fuse_qa_candidate_rows(
    primary: list[dict[str, object]],
    secondary: list[dict[str, object]],
    limit: int = 100,
    secondary_weight: float = .5,
    rrf_k: int = 60,
) -> list[dict[str, object]]:
    scores: dict[tuple[str, int], float] = {}
    rows: dict[tuple[str, int], dict[str, object]] = {}
    for ranking, weight in ((primary, 1.0), (secondary, secondary_weight)):
        for rank, row in enumerate(ranking, start=1):
            key = (str(row["video_id"]), int(row["frame_idx"]))
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            if key not in rows or ranking is primary:
                rows[key] = row
    protected = [
        (str(row["video_id"]), int(row["frame_idx"]))
        for row in primary[: min(10, limit)]
    ]
    protected_set = set(protected)
    ordered = protected + [
        key for key in sorted(scores, key=scores.get, reverse=True)
        if key not in protected_set
    ][: max(0, limit - len(protected))]
    return [{**rows[key], "rank": rank} for rank, key in enumerate(ordered, start=1)]


def compose_qa_hypothesis_candidates(
    primary: list[dict[str, object]],
    hypothesis_rankings: list[list[dict[str, object]]],
    reranked_union: list[dict[str, object]],
    limit: int = 100,
    primary_prefix: int = 3,
    hypothesis_depth: int = 10,
) -> list[dict[str, object]]:
    """Protect visual winners and fairly interleave possible-answer rankings."""
    output: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()

    def add(row: dict[str, object]) -> None:
        key = (str(row["video_id"]), int(row["frame_idx"]))
        if key not in seen and len(output) < limit:
            seen.add(key)
            output.append(dict(row))

    for row in primary[:primary_prefix]:
        add(row)
    for ranking in hypothesis_rankings:
        if ranking:
            add(ranking[0])
    for depth in range(1, max(1, hypothesis_depth)):
        for ranking in hypothesis_rankings:
            if depth < len(ranking):
                add(ranking[depth])
    for ranking in (reranked_union, primary, *hypothesis_rankings):
        for row in ranking:
            add(row)
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def expand_qa_context_rows(
    rows: list[dict[str, object]], metadata_path: Path,
    selected_count: int, radius: int = 1, limit: int = 100,
) -> list[dict[str, object]]:
    """Add frames Qwen actually saw, while preserving the selected Top-k prefix."""
    if not rows or selected_count <= 0 or radius <= 0:
        return rows[:limit]
    prefix = rows[: min(selected_count, len(rows), limit)]
    seen = {
        (str(row["video_id"]), int(row["frame_idx"])) for row in prefix
    }
    neighbors: list[dict[str, object]] = []
    with closing(sqlite3.connect(metadata_path)) as connection:
        for source_rank, row in enumerate(prefix, start=1):
            video_id = str(row["video_id"])
            keyframe_no = int(row["keyframe_no"])
            records = connection.execute(
                "SELECT video_id,keyframe_no,frame_idx,pts_time FROM keyframes "
                "WHERE video_id=? AND keyframe_no BETWEEN ? AND ? "
                "ORDER BY ABS(keyframe_no-?),keyframe_no",
                (video_id, keyframe_no - radius, keyframe_no + radius, keyframe_no),
            ).fetchall()
            for neighbor_video, neighbor_no, frame_idx, pts_time in records:
                key = (str(neighbor_video), int(frame_idx))
                if key in seen:
                    continue
                seen.add(key)
                neighbors.append({
                    **row,
                    "video_id": str(neighbor_video),
                    "keyframe_no": int(neighbor_no),
                    "frame_idx": int(frame_idx),
                    "pts_time": float(pts_time),
                    "context_of_rank": source_rank,
                })
    output = [*prefix, *neighbors]
    for row in rows[len(prefix):]:
        key = (str(row["video_id"]), int(row["frame_idx"]))
        if key not in seen:
            seen.add(key)
            output.append(row)
        if len(output) >= limit:
            break
    output = output[:limit]
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output

def rank_qa_answers(
    rows: list[dict[str, object]],
    predictions: list[tuple[int, str]],
    max_answers: int = 100,
    question: str = "",
) -> list[QAAnswer]:
    cleaned = [(index, clean_answer(answer)) for index, answer in predictions]
    frequencies = Counter(answer for _, answer in cleaned)
    first_seen = {answer: pos for pos, (_, answer) in enumerate(cleaned)}
    answers = sorted(
        frequencies,
        key=lambda answer: (-frequencies[answer], first_seen[answer]),
    )
    ranked: list[QAAnswer] = []
    seen: set[tuple[str, int, str]] = set()

    def add(row: dict[str, object], answer: str) -> None:
        key = (str(row["video_id"]), int(row["frame_idx"]), answer)
        if key not in seen and len(ranked) < max_answers:
            seen.add(key)
            ranked.append(QAAnswer(*key))

    for index, answer in cleaned:
        for alias in answer_aliases(question, answer):
            add(rows[index], alias)
    answer_by_source = {
        index + 1: answer_aliases(question, answer)[0]
        for index, answer in cleaned
    }
    for row in rows:
        source_rank = int(row.get("context_of_rank", 0))
        if source_rank in answer_by_source:
            add(row, answer_by_source[source_rank])
    for row in rows:
        for answer in answers:
            add(row, answer)
    return ranked


class QwenVLAnswerer:
    def __init__(self, device: str = "cuda", model_name: str = QWEN_VL_MODEL) -> None:
        self.device = device
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=192 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        if device == "cuda":
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization,
                device_map="cuda",
                dtype=torch.float16,
            ).eval()
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, dtype=torch.float32
            ).eval()

    def _generate(self, content: list[dict[str, object]], max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": content}]
        inputs = None
        generated = None
        try:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs, do_sample=False, max_new_tokens=max_new_tokens
                )
            prompt_length = inputs["input_ids"].shape[1]
            return self.processor.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True
            )[0]
        finally:
            generated = None
            inputs = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def answer(self, question: str, images: list[Image.Image]) -> str:
        content = [{"type": "image", "image": image} for image in images]
        content.append({
            "type": "text",
            "text": (
                "Answer the visual question using only the images. Return only "
                "the shortest exact answer, preferably 1-5 words, without an "
                f"explanation. Answer in the same language as the question. "
                f"Question: {question}"
            ),
        })
        return clean_answer(self._generate(content, 24))

    def score_temporal_sequence(
        self, events: list[str], image_groups: list[list[Image.Image]]
    ) -> int:
        content: list[dict[str, object]] = [{
            "type": "text",
            "text": (
                "You are verifying a chronological video-event sequence. "
                "For each target event, inspect its before/current/after images. "
                "Judge the action/state, chronological order, same scene, and "
                "same subject continuity. Return only one integer from 0 to 100; "
                "100 means every event and transition clearly matches."
            ),
        }]
        for index, (event, images) in enumerate(zip(events, image_groups), start=1):
            content.append({
                "type": "text", "text": f"Target event {index}: {event}"
            })
            content.extend({"type": "image", "image": image} for image in images)
        raw = self._generate(content, 8)
        match = re.search(r"\b(100|[1-9]?\d)\b", raw)
        if match is None:
            raise ValueError(f"VLM returned no temporal score: {raw[:80]}")
        return int(match.group(1))


def context_images(
    store: KeyframeStore, video_id: str, keyframe_no: int, radius: int = 1,
    crop_index: int | None = None,
) -> list[Image.Image]:
    images: list[Image.Image] = []
    for number in range(max(1, keyframe_no - radius), keyframe_no + radius + 1):
        try:
            payload = store.get_bytes(video_id, number)
        except (KeyError, FileNotFoundError):
            continue
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        images.append(image)
        if number == keyframe_no and crop_index is not None:
            images.append(crop_image(image, crop_index))
    return images


def search_qna(
    engine: MultilingualFaissEngine,
    store: KeyframeStore,
    answerer: QwenVLAnswerer,
    question: str,
    question_for_vlm: str | None = None,
    vlm_candidates: int = 12,
) -> list[QAAnswer]:
    rows = engine.search(
        question, 100, candidate_k=5000,
        per_video=3, min_time_gap=2.0, quality=False,
    )
    predictions: list[tuple[int, str]] = []
    for index, row in enumerate(rows[:vlm_candidates]):
        images = context_images(
            store, str(row["video_id"]), int(row["keyframe_no"])
        )
        if not images:
            continue
        try:
            predictions.append((
                index, answerer.answer(question_for_vlm or question, images)
            ))
        finally:
            for image in images:
                image.close()
    if not predictions:
        raise RuntimeError("VLM produced no Q&A predictions")
    return rank_qa_answers(rows, predictions, question=question)


def unload_retrieval_engine(engine: MultilingualFaissEngine) -> None:
    """Free retrieval model allocations before loading the 4-bit VLM."""
    if hasattr(engine, "model"):
        engine.model = None
    if getattr(engine, "hybrid", None) is not None and hasattr(engine.hybrid, "model"):
        engine.hybrid.model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def answer_rows(
    rows: list[dict[str, object]],
    store: KeyframeStore,
    answerer: QwenVLAnswerer,
    question: str,
    vlm_candidates: int,
) -> list[QAAnswer]:
    predictions: list[tuple[int, str]] = []
    for index, row in enumerate(rows[:vlm_candidates]):
        images = context_images(store, str(row["video_id"]), int(row["keyframe_no"]))
        if not images:
            continue
        try:
            predictions.append((index, answerer.answer(question, images)))
        finally:
            for image in images:
                image.close()
    if not predictions:
        raise RuntimeError("Qwen-VL produced no Q&A predictions")
    return rank_qa_answers(rows, predictions, question=question)

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve frames and answer visual questions")
    parser.add_argument("question", help="Question text or path to a UTF-8 query file")
    parser.add_argument("--question-en", help="Optional English translation for the VLM")
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--zip-dir", type=Path, default=Path(".."))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--vlm-candidates", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = Path(args.question)
    question = path.read_text(encoding="utf-8-sig").strip() if path.is_file() else args.question
    from web_app_vi import MultilingualFaissEngine as Engine
    engine = Engine(args.index_dir, args.device)
    store = KeyframeStore(args.zip_dir)
    rows = engine.search(
        question, 100, candidate_k=5000,
        per_video=3, min_time_gap=2.0, quality=False,
    )
    unload_retrieval_engine(engine)
    del engine
    answerer = QwenVLAnswerer(args.device)
    answers = answer_rows(
        rows, store, answerer, args.question_en or question, args.vlm_candidates
    )
    write_qa_submission(args.output, answers)
    print(f"Wrote {len(answers)} Q&A rows to {args.output}")


if __name__ == "__main__":
    main()
